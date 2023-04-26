import os
import argparse
import datetime
import time

from freezing import freezingModifications

from classifier import SentimentClassifier
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, \
	AutoModelForSequenceClassification
import torch
import dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch import nn
import torch.nn.functional as F
from math import ceil
from pyvacy import optim as optim_pyvacy
from pyvacy import analysis as pyvacy_analysis
from pyvacy import sampling
from pyvacy import analysis
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PromptEncoderConfig

parser = argparse.ArgumentParser(prog='script')
parser.add_argument('--train', action="store_true", help="Train new weights")
parser.add_argument('--path', type=str, help="Path of desired model/where you want results")
parser.add_argument('--paramF', action="store_true", help="Freeze subset of layers")
parser.add_argument('--layerF', action="store_true", help="Freeze subset of parameters")
parser.add_argument('--plF', action="store_true", help="Freeze subset of parameters and layers")
parser.add_argument('--epoch', type=int, help="Num Epochs")
parser.add_argument('--portion', type=int, help="Portion of layers/parameters to freeze")
parser.add_argument('--freeze', action="store_true", help="Freeze bert")
parser.add_argument('--evaluate', action="store_true", help="Evaluate existing weights")
parser.add_argument('--predict', default="", type=str, help="Predict sentiment on a given sentence")
parser.add_argument('--dp', action="store_true", help="use pyvacy")
parser.add_argument('--epsilon', action="store_true", help="find epsilon value for hardcoded inputs")
parser.add_argument('--sst', action="store_true", help="Load the SST dataset instead")
parser.add_argument('--lora', action="store_true", help="Use Lora to train the model")
parser.add_argument('--ptune', action="store_true", help="Use P-Tuning to train the model")

args = parser.parse_args()

if args.sst:
	PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
	class_names = ['negative', 'positive']
	MAX_LEN = 100

else:
	PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
	class_names = ['negative', "neutral", 'positive']
	MAX_LEN = 160

RANDOM_SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
L2_NORM_CLIP = 1.0
NOISE = 1.1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(out_dir, epochs):
	np.random.seed(RANDOM_SEED)
	torch.manual_seed(RANDOM_SEED)
	df_train, df_val, train_data_loader, val_data_loader = getData(True)

	print("Device", device)
	print(class_names)
	print(len(class_names), MAX_LEN, PRE_TRAINED_MODEL_NAME, args.sst)
	model = SentimentClassifier(len(class_names), PRE_TRAINED_MODEL_NAME, args)
	# model = model.to(device)
	if args.lora:
		model.bert.print_trainable_parameters()
		out_dir += "lora/"
	if args.ptune:
		out_dir += "ptune/"
		peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)
		model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME)
		model = get_peft_model(model, peft_config)
	model = model.to(device)
	if args.dp:
		optimizer = optim_pyvacy.DPAdam(
			params=model.parameters(),
			l2_norm_clip=L2_NORM_CLIP,
			noise_multiplier=NOISE,
			minibatch_size=BATCH_SIZE,
			microbatch_size=1,
			lr=LEARNING_RATE,
		)
	else:
		optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
	total_steps = len(train_data_loader) * epochs

	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=0,
		num_training_steps=total_steps
	)

	loss_fn = nn.CrossEntropyLoss().to(device)

	print("pre-freezing trainable params", sum(p.numel() for p in model.parameters() if p.requires_grad))

	model, out_dir = freezingModifications(args, model, out_dir)
	params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("post freezing trainable params", params)
	out_dir += datetime.datetime.now().strftime("%m-%d-%Y") + "/"
	best_accuracy = 0

	os.makedirs(out_dir, exist_ok=True)
	start_time = time.time()
	for epoch in tqdm(range(epochs)):
		print(f'Epoch {epoch + 1}/{epochs}')
		print('-' * 10)

		train_acc, train_loss = train_epoch(
			model,
			train_data_loader,
			loss_fn,
			optimizer,
			scheduler,
			len(df_train)
		)

		print(f'Train loss {train_loss} accuracy {train_acc}')

		val_acc, val_loss = eval_model(
			model,
			val_data_loader,
			loss_fn,
			len(df_val)
		)

		print(f'Val   loss {val_loss} accuracy {val_acc}')

		if val_acc > best_accuracy:
			torch.save(model.state_dict(), out_dir + 'best_model_state.bin')
			best_accuracy = val_acc
	total_time = time.time() - start_time
	return out_dir, total_time, params


def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
	model = model.train()
	if args.dp:
		minibatch_loader, microbatch_loader = sampling.get_data_loaders(
			BATCH_SIZE,
			1,
			ceil(n_examples / BATCH_SIZE)
		)
		trainset = getDPData()
		losses = []
		correct_predictions = 0
		for batch in tqdm(minibatch_loader(trainset)):
			input_ids = batch["input_ids"].to(device)
			attention_mask = batch["attention_mask"].to(device)
			targets = batch["targets"].to(device)
			for input_id_micro, attention_mask_micro, targets_micro in microbatch_loader(
					TensorDataset(input_ids, attention_mask, targets)):
				optimizer.zero_microbatch_grad()
				outputs = model(
					input_ids=input_id_micro,
					attention_mask=attention_mask_micro
				)
				_, preds = torch.max(outputs, dim=1)
				loss = loss_fn(outputs, targets_micro)
				correct_predictions += torch.sum(preds == targets_micro)

				losses.append(loss.item())
				loss.backward()
				optimizer.microbatch_step()
			# break
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()

		print(correct_predictions, n_examples)
		return correct_predictions.double() / n_examples, np.mean(losses)

	else:
		losses = []
		correct_predictions = 0

		iterator = tqdm(data_loader)

		for d in iterator:
			input_ids = d["input_ids"].to(device)
			attention_mask = d["attention_mask"].to(device)
			targets = d["targets"].to(device)
			outputs = model(
				input_ids=input_ids,
				attention_mask=attention_mask
			)
			if args.ptune:
				outputs = outputs['logits']
			_, preds = torch.max(outputs, dim=1)
			loss = loss_fn(outputs, targets)

			correct_predictions += torch.sum(preds == targets)
			# print("Preds", preds)
			# print("targets", targets)
			# print("correct_predictions", correct_predictions)
			# break
			losses.append(loss.item())

			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()

		return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, n_examples):
	model = model.eval()

	losses = []
	correct_predictions = 0

	with torch.no_grad():
		for d in data_loader:
			input_ids = d["input_ids"].to(device)
			attention_mask = d["attention_mask"].to(device)
			targets = d["targets"].to(device)

			outputs = model(
				input_ids=input_ids,
				attention_mask=attention_mask
			)

			if args.ptune:
				outputs = outputs['logits']

			_, preds = torch.max(outputs, dim=1)

			loss = loss_fn(outputs, targets)

			correct_predictions += torch.sum(preds == targets)
			losses.append(loss.item())

	return correct_predictions.double() / n_examples, np.mean(losses)


def getData(train):
	sst = True if args.sst else False
	if sst:
		df = pd.read_csv("sst2.csv")
	else:
		df = pd.read_csv("reviews.csv")
		df['sentiment'] = df.score.apply(dataset.to_sentiment)
	tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
	df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)

	if train:
		df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
		print("Training, validation, and test size sets", df_train.shape, df_val.shape, df_test.shape)
		train_data_loader = dataset.create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE, sst)
		val_data_loader = dataset.create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE, sst)
		return df_train, df_val, train_data_loader, val_data_loader

	else:
		test_data_loader = dataset.create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE, sst)
		return test_data_loader


def getDPData():
	sst = True if args.sst else False
	if sst:
		df = pd.read_csv("sst2.csv")
	else:
		df = pd.read_csv("reviews.csv")
		df['sentiment'] = df.score.apply(dataset.to_sentiment)
	tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
	df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
	return dataset.create_dataset(df_train, tokenizer, MAX_LEN, sst)


def evaluate(out_dir, total_time, epochs, paramNum):
	test_data_loader = getData(False)
	model = SentimentClassifier(len(class_names), PRE_TRAINED_MODEL_NAME, args)
	if args.ptune:
		peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)
		model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME)
		model = get_peft_model(model, peft_config)
	model.load_state_dict(torch.load(out_dir + 'best_model_state.bin'))
	model = model.to(device)
	y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
		model,
		test_data_loader
	)
	score = classification_report(y_test, y_pred, target_names=class_names)
	epsilon = findEpsilon(epochs)
	with open(out_dir + 'results.txt', 'w+') as f:
		if args.sst:
			f.write(f"Results for SST-2 GLUE dataset\n")
		else:
			f.write(f"Results for normal movie dataset\n")
		f.write(f"Total Time: {total_time} seconds, Epochs: {epochs}, Epsilon: {epsilon}\n")
		f.write(f"trainable_param: {paramNum}\n")
		f.write(f"LEARNING_RATE = {LEARNING_RATE}, L2_NORM_CLIP = {L2_NORM_CLIP}, NOISE = {NOISE}\n")
		f.write(score)

	print(score)


def get_predictions(model, data_loader):
	model = model.eval()

	review_texts = []
	predictions = []
	prediction_probs = []
	real_values = []

	with torch.no_grad():
		for d in data_loader:
			texts = d["review_text"]
			input_ids = d["input_ids"].to(device)
			attention_mask = d["attention_mask"].to(device)
			targets = d["targets"].to(device)

			outputs = model(
				input_ids=input_ids,
				attention_mask=attention_mask
			)
			_, preds = torch.max(outputs, dim=1)

			probs = F.softmax(outputs, dim=1)

			review_texts.extend(texts)
			predictions.extend(preds)
			prediction_probs.extend(probs)
			real_values.extend(targets)

	predictions = torch.stack(predictions).cpu()
	prediction_probs = torch.stack(prediction_probs).cpu()
	real_values = torch.stack(real_values).cpu()
	return review_texts, predictions, prediction_probs, real_values


def findEpsilon(epochs):
	if args.sst:
		batch_in_epoch = 886
		num_epoch = epochs
		epsilon = analysis.epsilon(14176, BATCH_SIZE, NOISE, BATCH_SIZE * batch_in_epoch * num_epoch, 1/60614)
		print("epsilon!", epsilon)
		return epsilon
	else:
		batch_in_epoch = 3789
		num_epoch = epochs
		epsilon = analysis.epsilon(60614, BATCH_SIZE, NOISE, BATCH_SIZE * batch_in_epoch * num_epoch)
		print("epsilon!", epsilon)
		return epsilon


if __name__ == '__main__':
	epochs = args.epoch or 10
	path = args.path or "results/"

	if args.train:
		if not args.dp:
			path += "noDP/"
		output_dir, seconds, paramNum = train(path, epochs)
		evaluate(output_dir, seconds, epochs, str(paramNum))

	if args.evaluate:
		evaluate(path, 10616, epochs, '294912')

	if args.epsilon:
		findEpsilon(epochs)
#
# if len(args.predict) > 0:
# 	print(predict(args.predict, args.path))
