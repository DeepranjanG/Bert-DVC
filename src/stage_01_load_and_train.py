from src.utils.all_utils import read_yaml
import pandas as pd
import argparse
import os
import shutil
from tqdm import tqdm
import logging
from transformers import AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from src.utils.dataset import SSTDataset
from src.utils.train import train
from src.utils.modeling import BertForSentimentClassification


logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def load_and_train(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    source_data = config["source_data"]
    train_data = os.path.join(source_data["data_dir"], source_data["train_data_file"])
    valid_data = os.path.join(source_data["data_dir"], source_data["dev_data_file"])

    BASE_MODEL_NAME = config["artifacts"]["BASE_MODEL_NAME"]
    
    train_set = SSTDataset(filename=train_data, maxlen=params["MAX_TRAIN"], tokenizer=tokenizer)
    valid_set = SSTDataset(filename=valid_data, maxlen=params["MAX_VALID"], tokenizer=tokenizer)
    
    train_loader = DataLoader(dataset=train_set, batch_size=params["BATCH_SIZE"], num_workers=params["NUM_WORKERS"])
    valid_loader = DataLoader(dataset=valid_set, batch_size=params["BATCH_SIZE"], num_workers=params["NUM_WORKERS"])


    auto_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)

    model = BertForSentimentClassification.from_pretrained(BASE_MODEL_NAME, config=auto_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(params=model.parameters(), lr=params["LEARNING_RATE"])

    epochs = params["EPOCHS"]

    train(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, val_loader=valid_loader, device=device, epochs=epochs)

    







if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(">>>>> stage one started <<<<<")
        load_and_train(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(">>>>> stage one completed! Data loaded <<<<<n")
    except Exception as e:
        logging.exception(e)
        raise e