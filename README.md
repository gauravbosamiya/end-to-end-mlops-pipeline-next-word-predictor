# Next Word Predictor - MLOps Project

This project is an end-to-end **Next Word Prediction** system built using **TensorFlow**, **Flask**, and **MLOps tools** such as **MLflow**, **DVC**, **Docker**, and **GitHub Actions**. It predicts the next few words in a sentence using LSTM model.

## Features

- Predicts the next 30 words for any input sentence.
- LSTM model trained on text data.
- Flask web app for UI.
- Model versioning and experiment tracking with MLflow & DagsHub.
- Dockerized for deployment.
- CI/CD pipeline integration with GitHub Actions.

## Tech Stack

- Python 3.10
- TensorFlow & Keras
- Flask
- MLflow + DagsHub
- Docker
- GitHub Actions (CI/CD)
- DVC (Data & model tracking)

## Running Locally

### 1. Clone the Repository
    git clone https://github.com/gauravbosamiya/end-to-end-mlops-pipeline-next-word-predictor.git
    cd end-to-end-mlops-pipeline-next-word-predictor

### 2. Create & Activate Virtual Environment (with Conda)
    conda create -n word python=3.10
    conda activate word

### 3. Install Dependencies
    pip install -r requirements.txt

### 4. Run the Flask App
    python flask_app/app.py
