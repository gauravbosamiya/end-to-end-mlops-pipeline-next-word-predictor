import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import yaml
import os
import dagshub
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner="gauravbosamiya"
repo_name="end-to-end-mlops-pipeline-next-word-predictor"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# ------------------------------------------------------------------------------------------------------
# MLflow & DAGsHub tracking
# MLFLOW_TRACKING_URI = "https://dagshub.com/gauravbosamiya/end-to-end-mlops-pipeline-next-word-predictor.mlflow"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# dagshub.init(repo_owner="gauravbosamiya", repo_name="end-to-end-mlops-pipeline-next-word-predictor", mlflow=True)
# ------------------------------------------------------------------------------------------------------

# Load parameters
def load_params(params_path):
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        raise

# Load data
def load_data(file_path_X, file_path_y):
    try:
        X_data = np.load(file_path_X)
        y_data = np.load(file_path_y)
        logging.info("Data loaded successfully.")
        return X_data, y_data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# Build LSTM Model
def build_model(vocab_size, sequence_length, embedding_dim, lstm_units, dropout):
    try:
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=sequence_length),
            LSTM(lstm_units, return_sequences=False),
            Dropout(dropout),
            Dense(vocab_size, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        logging.info(f"Model built with vocab size: {vocab_size}, sequence length: {sequence_length}")
        return model
    except Exception as e:
        logging.error(f"Error in model building: {e}")
        raise

# Train the model
def train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs):
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
        logging.error(f"Error in model training: {e}")
        raise

# Save the trained model locally
def save_model(model, file_path):
    try:
        model.save(file_path, save_format='h5')  # Explicit format
        logging.info(f"Model saved locally at {file_path}")
    except Exception as e:
        logging.error(f"Error saving the model: {e}")
        raise
    

def main():
    # mlflow.set_experiment("my-dvc-pipeline")

    # with mlflow.start_run() as run:
        try:
            # Define parameters
            params = load_params('params.yaml')
            embedding_dim = params['model_building']['embedding_dim']
            lstm_units = params['model_building']['lstm_units']
            dropout = params['model_building']['dropout']
            vocab_size = params['model_building']['vocab_size']
            epochs = params['model_building']['epochs']
            batch_size = params['model_building']['batch_size']
            
            # Log parameters
            # mlflow.log_params(params["model_building"])

            # Load data
            X_train, y_train = load_data("./data/processed/X_train.npy", "./data/processed/y_train.npy")
            X_val, y_val = load_data("./data/processed/X_val.npy", "./data/processed/y_val.npy")

            sequence_length = X_train.shape[1]
            
            # Build and compile the model
            model = build_model(vocab_size, sequence_length, embedding_dim, lstm_units, dropout)
            
            # Train the model
            history = train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs)
            
            # Log metrics for each epoch
            # for epoch in range(epochs):
            #     mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            #     mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            #     mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            #     mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            
            # Save the trained model locally
            model_path = "./models/LSTM_512.h5"
            save_model(model, model_path)

            # Log model as an MLflow artifact
            # mlflow.log_artifact(model_path)

            # Log the model to MLflow in the best practice way
            # mlflow.keras.log_model(model, "model")

            logging.info("Model Building Completed .")

        except Exception as e:
            logging.error(f"Error during model building and training: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    main()