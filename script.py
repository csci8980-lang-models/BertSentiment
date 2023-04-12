import os
import argparse
import random
from torch.utils.data import RandomSampler
# from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
import datetime
import torch

import transformers
from classifier import SentimentClassifier
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import dataset
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
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
PORTION = .9

parser = argparse.ArgumentParser(prog='script')
parser.add_argument('--train', action="store_true", help="Train new weights")
parser.add_argument('--paramF', action="store_true", help="Freeze subset of layers")
parser.add_argument('--layerF', action="store_true", help="Freeze subset of parameters")
parser.add_argument('-n', type=int, help="Dataset size")
parser.add_argument('--epoch', type=int, help="Num Epochs")
parser.add_argument('--freeze', action="store_true", help="Freeze bert")
parser.add_argument('--evaluate', action="store_true", help="Evaluate existing weights")
parser.add_argument('--predict', default="", type=str, help="Predict sentiment on a given sentence")
parser.add_argument('--path', default='weights/', type=str, help="Weights path")
parser.add_argument('--train-file', default='data/imdb_train.txt',
					type=str, help="IMDB train file. One sentence per line.")
parser.add_argument('--test-file', default='data/imdb_test.txt',
					type=str, help="IMDB train file. One sentence per line.")
args = parser.parse_args()

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
RANDOM_SEED = 42
class_names = ['negative', 'neutral', 'positive']
MAX_LEN = 160
BATCH_SIZE = 16
EPOCHS = 10



def train(train_file, epochs, output_dir, n):
	np.random.seed(RANDOM_SEED)
	torch.manual_seed(RANDOM_SEED)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	df = getData()
	tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
	df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
	df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
	print("Training, validation, and test size sets", df_train.shape, df_val.shape, df_test.shape)

	train_data_loader = dataset.create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
	val_data_loader = dataset.create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
	test_data_loader = dataset.create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

	print("Device", device)
	model = SentimentClassifier(len(class_names))
	model = model.to(device)
	optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
	total_steps = len(train_data_loader) * EPOCHS

	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=0,
		num_training_steps=total_steps
	)

	loss_fn = nn.CrossEntropyLoss().to(device)

	if args.freeze:
		output_dir = output_dir + "freeze/"
		for layer in model.bert.encoder.layer:
			for param in layer.parameters():
				param.requires_grad = False

	if args.layerF:
		output_dir = output_dir + "layerF/"
		layers = [model.fc]
		for layer in model.bert.encoder.layer:
			layers.append(layer)

		count = int(len(layers) * PORTION)
		layers = random.sample(layers, count)

		for layer in layers:
			for param in layer.parameters():
				param.requires_grad = False

	if args.paramF:
		output_dir = output_dir + "paramF/"
		parameters = []
		for layer in model.bert.encoder.layer:
			for param in layer.parameters():
				if param.requires_grad:
					parameters.append(param)

		for param in model.parameters():
			if param.requires_grad:
				parameters.append(param)

		count = int(len(parameters) * PORTION)
		subset = random.sample(parameters, count)

		for param in subset:
			param.requires_grad = False
	history = defaultdict(list)
	best_accuracy = 0

	for epoch in range(EPOCHS):

		print(f'Epoch {epoch + 1}/{EPOCHS}')
		print('-' * 10)

		train_acc, train_loss = train_epoch(
			model,
			train_data_loader,
			loss_fn,
			optimizer,
			device,
			scheduler,
			len(df_train)
		)

		print(f'Train loss {train_loss} accuracy {train_acc}')

		val_acc, val_loss = eval_model(
			model,
			val_data_loader,
			loss_fn,
			device,
			len(df_val)
		)

		print(f'Val   loss {val_loss} accuracy {val_acc}')
		print()

		history['train_acc'].append(train_acc)
		history['train_loss'].append(train_loss)
		history['val_acc'].append(val_acc)
		history['val_loss'].append(val_loss)

		if val_acc > best_accuracy:
			torch.save(model.state_dict(), 'best_model_state.bin')
			best_accuracy = val_acc


def train_epoch(
		model,
		data_loader,
		loss_fn,
		optimizer,
		device,
		scheduler,
		n_examples
):
	model = model.train()

	losses = []
	correct_predictions = 0
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

		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()
		scheduler.step()
		optimizer.zero_grad()

	return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
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
	print("DATA FRAME Head", df.head())

	print("DF Shape:", df.shape)

	print(df.hist(column="score"))

	df['sentiment'] = df.score.apply(dataset.to_sentiment)
	return df

# def evaluate(test_file, model_dir, n):
# 	n = int(n / 2)
# 	predictor = SentimentBERT()
# 	predictor.load(model_dir=model_dir)
#
# 	dt = SentimentDataset(predictor.tokenizer)
# 	dataloader = dt.prepare_dataloader(test_file, n)
# 	score = predictor.evaluate(dataloader, n)
# 	print("SCORE!")
# 	filename = model_dir + datetime.datetime.now().strftime("%m-%d-%Y %H %M") + ".txt"
# 	with open(filename, 'w+') as f:
# 		f.write(score)
# 	print(score)


# def predict(text, model_dir):
# 	predictor = SentimentBERT()
# 	predictor.load(model_dir=model_dir)
#
# 	dt = SentimentDataset(predictor.tokenizer)
# 	dataloader = dt.prepare_dataloader_from_examples([(text, -1)], sampler=None)  # text and a dummy label
# 	result = predictor.predict(dataloader)
#
# 	return "Positive" if result[0] == 0 else "Negative"


if __name__ == '__main__':
	train("train_file", 10, "weights", 5000)
	# epochs = args.epoch or 20
	# n = args.n or 25000
	# path = args.path or "weights/"
	# if args.train:
	# 	os.makedirs(args.path, exist_ok=True)
	# 	train(args.train_file, epochs, path, n)
	#
	# if args.evaluate:
	# 	evaluate(args.test_file, path, n)
	#
	# if len(args.predict) > 0:
	# 	print(predict(args.predict, args.path))