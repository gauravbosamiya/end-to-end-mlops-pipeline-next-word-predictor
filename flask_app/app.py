import numpy as np 
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.logger import logging
import mlflow
import dagshub
from flask import Flask, render_template, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")
import time
import pickle


mlflow.set_tracking_uri("https://dagshub.com/gauravbosamiya/end-to-end-mlops-pipeline-next-word-predictor.mlflow")
dagshub.init(repo_owner="gauravbosamiya", repo_name="end-to-end-mlops-pipeline-next-word-predictor", mlflow=True)



registry = CollectorRegistry()

# Define your custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

app = Flask(__name__)


model_name = "my_model"
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    
    latest_version = client.get_latest_versions(model_name, stages=["Staging"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["Production"])

    return latest_version[0].version if latest_version else None

model_version = get_latest_model_version(model_name)
model_uri = f'models:/{model_name}/{model_version}'
print(f"Fetching model from: {model_uri}")
model = mlflow.keras.load_model(model_uri)
LSTM = load_model("./models/LSTM_512.h5")



def load_word_index(filepath):
    word_index = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                word, index = line.split(':')
                word_index[word.strip()] = int(index.strip())
    return word_index

@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response




@app.route("/predict", methods=["POST"])
def predict(num_words=30):
    with open("./data/interim/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    text = request.form["text"]
    for i in range(num_words):
        token_text = tokenizer.texts_to_sequences([text])[0]
        
        padded_token_text = pad_sequences([token_text],maxlen=177, padding="pre")
        
        predicted_word = None
        pos = np.argmax(model.predict(padded_token_text,verbose=0))
        
        for word, index in tokenizer.word_index.items():
            if index==pos:
                predicted_word = word
                break
            
        if predicted_word is None:
            break

        text = text + " " + predicted_word
        
        
    PREDICTION_COUNT.labels(prediction=str(predicted_word)).inc()

    # Measure latency
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        
    return render_template("index.html", predicted_text=text)



@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(debug=True) # for local use
    # app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker




        
    


    