import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

class BertForSentimentClassification(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.bert = BertModel(config)
		self.cls_layer = nn.Linear(config.hidden_size, 1)

	def forward(self, input_ids, attention_mask):
		reps, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		cls_reps = reps[:, 0]
		logits = self.cls_layer(cls_reps)
		return logits
