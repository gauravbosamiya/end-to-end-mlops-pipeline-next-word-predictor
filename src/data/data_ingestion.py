import numpy as np 
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging
import requests

def load_params(param_path):
    try:
        with open(param_path,"r") as file:
            params = yaml.safe_load(file)
        logging.debug("parameters retrieved from %s", param_path)
        return params
    except FileNotFoundError:
        logging.error("File not found: %s", param_path)
        raise
    except yaml.YAMLError as e:
        logging.error("YAML Error: %s",e)
        raise
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        raise


def load_data(data_url):
    try:
        response = requests.get(data_url)
        if response.status_code == 200:
            text = response.text.strip()  
            logging.info("Data successfully loaded from %s", data_url)
            return text.split("\n")  # Return list of lines
        else:
            logging.error("Failed to load data. Status code: %d", response.status_code)
            raise Exception(f"Failed to load data. HTTP Status: {response.status_code}")
    except requests.RequestException as e:
        logging.error("Error while making request: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error while loading file: %s", e)
        raise
    
def save_data(train_data,data_path):
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        
        print("Saving train data...")  
        with open(os.path.join(raw_data_path, "train.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(train_data))
        logging.debug("Train data saved to %s", raw_data_path)
        print("Data saved successfully.") 
        
    except Exception as e:
        logging.error("Unexpected error while saving data: %s", e)
        raise

def main():
    try:
        train_data = load_data(data_url="https://raw.githubusercontent.com/gauravbosamiya/Datasets/refs/heads/main/text.txt")
        save_data(train_data,data_path='./data')
    except Exception as e:
        logging.error("Failed to complete the data ingestion process: %s", e)
        print(f"Error: {e}")

if __name__ =="__main__":
    main()

