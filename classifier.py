
from transformers import BertModel

from torch import nn
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PromptEncoderConfig

class SentimentClassifier(nn.Module):

	def __init__(self, n_classes, model_name, args):
		super(SentimentClassifier, self).__init__()
		self.bert = BertModel.from_pretrained(model_name)
		if args.lora:
			peft_config = LoraConfig(
				inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
			)
			self.bert = get_peft_model(self.bert, peft_config)
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