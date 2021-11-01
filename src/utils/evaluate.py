import torch
import tqdm


def get_accuracy_from_logits(logits, labels):
	#Get a tensor of shape [B, 1, 1] with probabilities that the sentiment is positive
	probs = torch.sigmoid(logits.unsqueeze(-1))
	#Convert probabilities to predictions, 1 being positive and 0 being negative
	soft_probs = (probs > 0.5).long()
	#Check which predictions are the same as the ground truth and calculate the accuracy
	acc = (soft_probs.squeeze() == labels).float().mean()
	return acc

def evaluate(model, criterion, dataloader, device):
    model.eval()
    mean_acc, mean_loss, count = 0, 0, 0
    print(dataloader)
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            logits = model(input_ids, attention_mask)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1
    return mean_acc / count, mean_loss / count