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
from src.utils.evaluate import evaluate



logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def model_evaluation(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    source_data = config["source_data"]
    valid_data = os.path.join(source_data["data_dir"], source_data["dev_data_file"])

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