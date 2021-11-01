
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from tqdm import tqdm, trange
from transformers import AutoConfig, AutoTokenizer
# from src.utils.all_utils import evaluate


def train(model, criterion, optimizer, train_loader, val_loader, device, epochs):
	best_acc = 0
	for epoch in trange(epochs, desc="Epoch"):
		model.train()
		for i, (input_ids, attention_mask, labels) in enumerate(tqdm(iterable=train_loader, desc="Training")):
			optimizer.zero_grad()  
			input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
			logits = model(input_ids=input_ids, attention_mask=attention_mask)
			loss = criterion(input=logits.squeeze(-1), target=labels.float())
			loss.backward()
			optimizer.step()
		# val_acc, val_loss = evaluate(model=model, criterion=criterion, dataloader=val_loader, device=device)
		# print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(epoch, val_acc, val_loss))
		# if val_acc > best_acc:
		# 	print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
		# 	best_acc = val_acc
		model.save_pretrained(save_directory=f'artifacts/my_model/')
		# config.save_pretrained(save_directory=f'artifacts/my_model/')
		# tokenizer.save_pretrained(save_directory=f'artifacts/my_model/')