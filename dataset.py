import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

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

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class GPReviewDataset(Dataset):

	def __init__(self, reviews, targets, tokenizer, max_len):
		self.reviews = reviews
		self.targets = targets
		self.tokenizer = tokenizer
		self.max_len = max_len

	def __len__(self):
		return len(self.reviews)

	def __getitem__(self, item):
		review = str(self.reviews[item])
		target = self.targets[item]

		encoding = self.tokenizer.encode_plus(
			review,
			add_special_tokens=True,
			max_length=self.max_len,
			return_token_type_ids=False,
			pad_to_max_length=True,
			return_attention_mask=True,
			return_tensors='pt',
		)

		return {
			'review_text': review,
			'input_ids': encoding['input_ids'].flatten(),
			'attention_mask': encoding['attention_mask'].flatten(),
			'targets': torch.tensor(target, dtype=torch.long)
		}


def to_sentiment(rating):
	rating = int(rating)
	if rating <= 2:
		return 0
	elif rating == 3:
		return 1
	else:
		return 2


def create_data_loader(df, tokenizer, max_len, batch_size):
	ds = GPReviewDataset(
		reviews=df.content.to_numpy(),
		targets=df.sentiment.to_numpy(),
		tokenizer=tokenizer,
		max_len=max_len
	)

	return DataLoader(
		ds,
		batch_size=batch_size,
	)
