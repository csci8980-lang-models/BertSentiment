import transformers
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

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("reviews.csv")
print("DATA FRAME Head", df.head())

print("DF Shape:", df.shape)

print(df.hist(column="score"))

df['sentiment'] = df.score.apply(dataset.to_sentiment)
class_names = ['negative', 'neutral', 'positive']

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'

tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f' Sentence: {sample_txt}')
print(f'   Tokens: {tokens}')
print(f'Token IDs: {token_ids}')

encoding = tokenizer.encode_plus(
	sample_txt,
	max_length=32,
	add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
	return_token_type_ids=False,
	pad_to_max_length=True,
	return_attention_mask=True,
	return_tensors='pt',  # Return PyTorch tensors
)

print(encoding.keys())

print(len(encoding['input_ids'][0]))
print(encoding['input_ids'][0])

print(len(encoding['attention_mask'][0]))
print(encoding['attention_mask'])

MAX_LEN = 160

df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

print("Training, validation, and test size sets", df_train.shape, df_val.shape, df_test.shape)

BATCH_SIZE = 16

train_data_loader = dataset.create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = dataset.create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = dataset.create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

data = next(iter(train_data_loader))
data.keys()

print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)


class SentimentClassifier(nn.Module):

	def __init__(self, n_classes):
		super(SentimentClassifier, self).__init__()
		self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
		self.drop = nn.Dropout(p=0.3)
		self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

	def forward(self, input_ids, attention_mask):
		_, pooled_output = self.bert(
			input_ids=input_ids,
			attention_mask=attention_mask,
			return_dict=False
		)
		output = self.drop(pooled_output)
		return self.out(output)


print("Device", device)
model = SentimentClassifier(len(class_names))
model = model.to(device)

input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

print(input_ids.shape)  # batch size x seq length
print(attention_mask.shape)  # batch size x seq length

F.softmax(model(input_ids, attention_mask), dim=1)

EPOCHS = 10

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
	optimizer,
	num_warmup_steps=0,
	num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)


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
