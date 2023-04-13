import os
import argparse
import random
from torch.utils.data import RandomSampler
# from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
import datetime
import time
import torch

from freezing import freezingModifications

import transformers
from classifier import SentimentClassifier
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from tqdm import tqdm

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

BERT_MODEL = 'bert-base-uncased'
NUM_LABELS = 2  # negative and positive reviews

parser = argparse.ArgumentParser(prog='script')
parser.add_argument('--train', action="store_true", help="Train new weights")
parser.add_argument('--path', action="store_true", help="Path of desired model/where you want results")
parser.add_argument('--paramF', action="store_true", help="Freeze subset of layers")
parser.add_argument('--layerF', action="store_true", help="Freeze subset of parameters")
parser.add_argument('--plF', action="store_true", help="Freeze subset of parameters and layers")
parser.add_argument('--epoch', type=int, help="Num Epochs")
parser.add_argument('--freeze', action="store_true", help="Freeze bert")
parser.add_argument('--evaluate', action="store_true", help="Evaluate existing weights")
parser.add_argument('--predict', default="", type=str, help="Predict sentiment on a given sentence")
args = parser.parse_args()

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
RANDOM_SEED = 42
class_names = ['negative', 'neutral', 'positive']
MAX_LEN = 160
BATCH_SIZE = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(out_dir, epochs):
	np.random.seed(RANDOM_SEED)
	torch.manual_seed(RANDOM_SEED)
	df_train, df_val, train_data_loader, val_data_loader = getData()

	print("Device", device)
	model = SentimentClassifier(len(class_names))
	model = model.to(device)
	optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
	total_steps = len(train_data_loader) * epochs

	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=0,
		num_training_steps=total_steps
	)

	loss_fn = nn.CrossEntropyLoss().to(device)

	for layer in model.bert.encoder.layer:
		for param in layer.parameters():
			param.requires_grad = True

	print("trainable params", sum(p.numel() for p in model.parameters() if p.requires_grad))

	model, out_dir = freezingModifications(args, model, out_dir)
	print("trainable params", sum(p.numel() for p in model.parameters() if p.requires_grad))
	out_dir += datetime.datetime.now().strftime("%m-%d-%Y") + "/"
	best_accuracy = 0

	os.makedirs(out_dir, exist_ok=True)
	start_time = time.time()
	for epoch in range(epochs):

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
	return out_dir, total_time


def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
	model = model.train()

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

		_, preds = torch.max(outputs, dim=1)
		loss = loss_fn(outputs, targets)

		correct_predictions += torch.sum(preds == targets)
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
			_, preds = torch.max(outputs, dim=1)

			loss = loss_fn(outputs, targets)

			correct_predictions += torch.sum(preds == targets)
			losses.append(loss.item())

	return correct_predictions.double() / n_examples, np.mean(losses)


def getData():
	df = pd.read_csv("reviews.csv")
	df['sentiment'] = df.score.apply(dataset.to_sentiment)
	tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
	df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
	print("Training size vs. predicted", len(df_train), len(df) * .9)
	df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
	print("Training, validation, and test size sets", df_train.shape, df_val.shape, df_test.shape)

	train_data_loader = dataset.create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
	val_data_loader = dataset.create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

	return df_train, df_val, train_data_loader, val_data_loader


def evaluate(out_dir, total_time):
	df = getData()
	tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
	df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
	df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
	print("Training, validation, and test size sets", df_train.shape, df_val.shape, df_test.shape)

	test_data_loader = dataset.create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = SentimentClassifier(len(class_names))
	model.load_state_dict(torch.load(out_dir + 'best_model_state.bin'))
	model = model.to(device)
	y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
		model,
		test_data_loader
	)
	score = classification_report(y_test, y_pred, target_names=class_names)
	with open(out_dir + 'results.txt', 'w+') as f:
		f.write(f"Total Time: {total_time} seconds")
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


if __name__ == '__main__':
	epochs = args.epoch or 10
	path = args.path or "results/"
	if args.train:
		output_dir, seconds = train(path, epochs)
		evaluate(output_dir, seconds)

	if args.evaluate:
		evaluate(path, 0)
#
# if len(args.predict) > 0:
# 	print(predict(args.predict, args.path))
