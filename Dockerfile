FROM python:3.10-slim

WORKDIR /app

COPY flask_app/requirements.txt /app/requirements.txt

# Install dependencies early to use Docker cache
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY flask_app/ /app/

COPY models/LSTM_512.h5 /app/models/LSTM_512.h5
COPY data/interim/tokenizer.pkl /app/data/interim/tokenizer.pkl
# COPY data/interim/padded_sequences.npy /app/data/interim/padded_sequences.npy

# RUN pip install -r requirements.txt

EXPOSE 5000

#local
CMD ["python", "app.py"]  

#Prod
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]