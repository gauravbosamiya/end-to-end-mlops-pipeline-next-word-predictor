import numpy as np
import json
import tensorflow as tf
import os
import mlflow
import dagshub
from tensorflow.keras.models import load_model
from src.logger import logging
import warnings
import yaml

warnings.filterwarnings("ignore")

# Configure MLflow and Dagshub
mlflow.set_tracking_uri("https://dagshub.com/gauravbosamiya/end-to-end-mlops-pipeline-next-word-predictor.mlflow")
dagshub.init(repo_owner="gauravbosamiya", repo_name="end-to-end-mlops-pipeline-next-word-predictor", mlflow=True)


def load_params(params_path):
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        raise

def load_data(file_path_X, file_path_y):    
    """Load evaluation data from numpy files."""
    try:
        X_test = np.load(file_path_X)
        y_test = np.load(file_path_y)
        logging.info("Evaluation data loaded successfully.")
        return X_test, y_test
    except Exception as e:
        logging.error("Error loading evaluation data: %s", e)
        raise

def compute_perplexity_sample(model, X_test, y_test, sample_size):
    """Compute perplexity using a small sample of the test set."""
    try:
        sample_size = min(sample_size, len(X_test))  
        X_sample = X_test[:sample_size]
        y_sample = y_test[:sample_size]

        predictions = model.predict(X_sample, verbose=0)
        y_true_indices = np.argmax(y_sample, axis=1)
        predicted_probs = np.array([pred[i] for pred, i in zip(predictions, y_true_indices)])

        log_probs = -np.log(predicted_probs + 1e-9)  
        perplexity = np.exp(np.mean(log_probs))

        logging.info("Perplexity on sample calculated: %f", perplexity)
        return float(perplexity)  # Convert to Python float
    except Exception as e:
        logging.error("Error computing sample perplexity: %s", e)
        raise

def save_metrics(metrics, file_path):
    """Save evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  
        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logging.info("Metrics saved to: %s", file_path)
    except Exception as e:
        logging.error("Error saving metrics: %s", e)
        raise

def save_model_info(run_id, model_path, file_path):
    """Save model information (run ID and model path) to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.info("Model info saved to %s", file_path)
    except Exception as e:
        logging.error("Error saving model info: %s", e)
        raise

# def main():
#     mlflow.set_experiment("my-dvc-pipeline")

#     with mlflow.start_run() as run:
#         try:
#             params = load_params('params.yaml')
#             sample_size = params['model_evaluation']['sample_size']
            
#             mlflow.log_params(params["model_evaluation"])
            
#             X_test, y_test = load_data("./data/processed/X_val.npy", "./data/processed/y_val.npy")
#             model_path = "./models/LSTM_512.h5"

#             # Load and evaluate model
#             model = load_model(model_path)
#             perplexity = compute_perplexity_sample(model, X_test, y_test, sample_size)

#             metrics = {"perplexity": perplexity}

#             # Log metric to MLflow (after ensuring it's a float)
#             mlflow.log_metric("perplexity", float(perplexity))

#             # Save metrics JSON file
#             save_metrics(metrics, "./reports/evaluation_metrics.json")
#             mlflow.log_artifact("./reports/evaluation_metrics.json")

#             # Save model info
#             save_model_info(run.info.run_id, "model", './reports/model_info.json')
#             mlflow.log_artifact("./reports/model_info.json")

#             mlflow.keras.log_model(model, "model")

#             logging.info("Model evaluation and MLflow logging completed successfully.")
#         except Exception as e:
#             logging.error("Evaluation process failed: %s", e)
#             print(f"Error: {e}")


def main():
    mlflow.set_experiment("my-dvc-pipeline")

    with mlflow.start_run() as run:
        try:
            params = load_params('params.yaml')
            print("Loaded Parameters:", params)  # Debugging step
            
            if "model_evaluation" not in params:
                raise KeyError("Key 'model_evaluation' not found in params.yaml")

            sample_size = params["model_evaluation"]["sample_size"]
            mlflow.log_params(params["model_evaluation"])

            X_test, y_test = load_data("./data/processed/X_val.npy", "./data/processed/y_val.npy")
            model_path = "./models/LSTM_512.h5"

            model = load_model(model_path)
            perplexity = compute_perplexity_sample(model, X_test, y_test, sample_size)

            metrics = {"perplexity": perplexity}

            mlflow.log_metric("perplexity", float(perplexity))
            save_metrics(metrics, "./reports/evaluation_metrics.json")
            mlflow.log_artifact("./reports/evaluation_metrics.json")

            save_model_info(run.info.run_id, "model", './reports/model_info.json')
            mlflow.log_artifact("./reports/model_info.json")

            mlflow.keras.log_model(model, "model")

            logging.info("Model evaluation and MLflow logging completed successfully.")
        except Exception as e:
            logging.error("Evaluation process failed: %s", e)
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
