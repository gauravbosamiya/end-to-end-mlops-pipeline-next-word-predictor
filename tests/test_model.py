# load test + signature test + performance test

import unittest
import mlflow
import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner="gauravbosamiya"
        repo_name="end-to-end-mlops-pipeline-next-word-predictor"

        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.model = mlflow.keras.load_model(cls.new_model_uri)
        cls.lstm_model = load_model("./models/LSTM_512.h5")

        with open("./data/interim/tokenizer.pkl", "rb") as f:
            cls.tokenizer = pickle.load(f)
        cls.maxlen = 177
        
        # Load data for performance tests
        cls.X_train = np.load("./data/processed/X_train.npy")
        cls.y_train = np.load("./data/processed/y_train.npy")
        cls.X_val = np.load("./data/processed/X_val.npy")
        cls.y_val = np.load("./data/processed/y_val.npy")


    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model, "MLflow model should be loaded")
        self.assertIsNotNone(self.lstm_model, "Local LSTM model should be loaded")


    def test_model_signature(self):
        text = "Machine learning is"
        token_sequence = self.tokenizer.texts_to_sequences([text])[0]
        padded_input = pad_sequences([token_sequence], maxlen=self.maxlen, padding="pre")
        prediction = self.model.predict(padded_input, verbose=0)


        self.assertEqual(prediction.shape[0], 1, "Prediction should return one sequence")
        self.assertEqual(len(prediction.shape), 2, "Prediction output should be 2D (batch_size, vocab_size)")


    def test_generate_word(self):
        text = "Deep learning"
        original_length = len(text.split())

        for _ in range(5):
            token_sequence = self.tokenizer.texts_to_sequences([text])[0]
            padded = pad_sequences([token_sequence], maxlen=self.maxlen, padding="pre")
            prediction = np.argmax(self.model.predict(padded, verbose=0))

            word = None
            for w, index in self.tokenizer.word_index.items():
                if index == prediction:
                    word = w
                    break
            if not word:
                break

            text += " " + word

        self.assertGreater(len(text.split()), original_length, "Generated text should be longer than original")

    
    def test_model_performance_threshold(self):
        """Ensure model performance is above a defined threshold  70%."""
        train_loss, train_acc = self.lstm_model.evaluate(self.X_train, self.y_train, verbose=0)
        val_loss, val_acc = self.lstm_model.evaluate(self.X_val, self.y_val, verbose=0)

        self.assertGreaterEqual(train_acc, 0.70, "Train accuracy should be at least 0.05%")
        self.assertGreaterEqual(val_acc, 0.70, "Validation accuracy should be at least 0.05%")
        self.assertLessEqual(train_loss, 0.70, "Train loss should be less than or equal to 10%")
        self.assertLessEqual(val_loss, 0.70, "Validation loss should be less than or equal to 10%")



if __name__ == "__main__":
    unittest.main()