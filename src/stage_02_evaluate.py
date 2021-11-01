from src.utils.all_utils import read_yaml
import argparse
import os
import shutil
from tqdm import tqdm
import logging
from transformers import AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from src.utils.dataset import SSTDataset
from src.utils.train import train
from src.utils.modeling import BertForSentimentClassification
# from src.utils.evaluate import evaluate
import torch
import tqdm

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


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
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            logits = model(input_ids, attention_mask)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1
    return mean_acc / count, mean_loss / count



tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def model_evaluation(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    source_data = config["source_data"]
    valid_data = os.path.join(source_data["data_dir"], source_data["dev_data_file"])
    print(valid_data)

    BASE_MODEL_NAME = config["artifacts"]["BASE_MODEL_NAME"]

    auto_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)

    #Tokenizer for the desired transformer model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    #Create the model with the desired transformer model
    model = BertForSentimentClassification.from_pretrained(BASE_MODEL_NAME)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #Takes as the input the logits of the positive class and computes the binary cross-entropy 
    criterion = nn.BCEWithLogitsLoss()

    val_set = SSTDataset(filename=valid_data, maxlen=params["MAX_VALID"], tokenizer=tokenizer)
    print(val_set)
    val_loader = DataLoader(dataset=val_set, batch_size=params["BATCH_SIZE"], num_workers=params["NUM_WORKERS"])

    val_acc, val_loss = evaluate(model=model, criterion=criterion, dataloader=val_loader, device=device)
    print("Validation Accuracy : {}, Validation Loss : {}".format(val_acc, val_loss))




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(">>>>> stage two started <<<<<")
        model_evaluation(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(">>>>> stage two completed! Evaluation Completed <<<<<n")
    except Exception as e:
        logging.exception(e)
        raise e