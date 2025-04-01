import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import yaml
import os
from src.logger import logging

# Load parameters from params.yaml (configuration file)
def load_params(params_path):
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.debug("Parameters loaded from %s", params_path)
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

# Load data (X_train, y_train, X_val, y_val)
def load_data(file_path_X, file_path_y):
    try:
        X_data = np.load(file_path_X)
        y_data = np.load(file_path_y)
        logging.info("Data loaded successfully.")
        return X_data, y_data
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise

# Build LSTM Model for Next Word Prediction
def build_model(vocab_size, sequence_length, embedding_dim=100, lstm_units=256):
    try:
        model = Sequential()
        # Embedding Layer
        model.add(Embedding(vocab_size, embedding_dim, input_length=sequence_length))

        # LSTM Layer
        model.add(LSTM(lstm_units, return_sequences=False))
        
        # Dropout Layer (to reduce overfitting)
        model.add(Dropout(0.2))
        
        # Dense Layer with Softmax to predict the next word
        model.add(Dense(vocab_size, activation='softmax'))
        
        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        
        logging.info(f"Model built with vocab size: {vocab_size}, sequence length: {sequence_length}")
        return model
    except Exception as e:
        logging.error("Error in model building: %s", e)
        raise

# Train the model
def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=5):
    try:
        history = model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(X_val, y_val),
            verbose=1
        )
        logging.info("Model training completed.")
        return history
    except Exception as e:
        logging.error("Error in model training: %s", e)
        raise

# Save the trained model
def save_model(model, file_path):
    try:
        model.save(file_path)
        logging.info(f"Model saved to {file_path}")
    except Exception as e:
        logging.error("Error saving the model: %s", e)
        raise

# Main function
def main():
    try:
        # Load params from params.yaml (configuration file)
        # params = load_params('params.yaml')
        
        # Load data
        X_train, y_train = load_data("./data/processed/X_train.npy", "./data/processed/y_train.npy")
        X_val, y_val = load_data("./data/processed/X_val.npy", "./data/processed/y_val.npy")
        
        # Get vocab size (number of unique words in the vocabulary)
        vocab_size = 883 
        sequence_length = X_train.shape[1]
        
        # Build and compile the model
        model = build_model(vocab_size, sequence_length)
        
        # Train the model
        train_model(model, X_train, y_train, X_val, y_val)
        
        # Save the trained model
        save_model(model, './models/next_word_prediction_model.h5')
        
    except Exception as e:
        logging.error("Error during model building and training: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
