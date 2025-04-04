import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from src.logger import logging
import pickle
import yaml
from sklearn.model_selection import train_test_split

def load_params(params_path):
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logging.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logging.error("YAML Error: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        raise

def load_data(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)  # Loading padded sequences
        logging.info("Padded sequences loaded from %s", file_path)
        return data
    except Exception as e:
        logging.error("Unexpected error while loading file: %s", e)
        raise

def split_X_y_from_padded_sequences(padded_sequences, num_classes):
    """Splits the padded sequences into X (input) and y (target word)."""
    try:
        X = padded_sequences[:, :-1]  
        y = padded_sequences[:, -1]
        
        # One-hot encode y (target word)
        y = to_categorical(y, num_classes=num_classes) 
        
        logging.info(f"X and y split completed. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    except Exception as e:
        logging.error("Error during splitting data: %s", e)
        raise

def save_data(data, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True) 
        np.save(file_path, data) 
        logging.info("Data saved to %s", file_path)
    except Exception as e:
        logging.error("Unexpected error occurred while saving the data: %s", e)
        raise

def main():
    try:
        logging.info("Feature engineering for Next Word Prediction started...")
        
        # Load parameters
        params = load_params('params.yaml')
        test_size = params['feature_engineering']['test_size']
        num_classes = params['feature_engineering']['num_classes']
        random_state = params['feature_engineering']['random_state']
        
        
        # Load padded sequences
        padded_sequences = load_data('./data/interim/padded_sequences.npy')
        
        # Split into X and y
        X, y = split_X_y_from_padded_sequences(padded_sequences, num_classes)
        
        logging.info(f"Data after splitting into X and y - X shape: {X.shape}, y shape: {y.shape}")
        
        # Optionally split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logging.info("Data split into training and validation sets.")
        
        # Save train and validation sets inside the processed folder
        save_data(X_train, './data/processed/X_train.npy')
        save_data(X_val, './data/processed/X_val.npy')
        save_data(y_train, './data/processed/y_train.npy')
        save_data(y_val, './data/processed/y_val.npy')

        logging.info("Feature engineering for Next Word Prediction completed!")
    
    except Exception as e:
        logging.error("Failed to complete the feature engineering process: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
