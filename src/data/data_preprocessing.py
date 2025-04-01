import re
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.logger import logging
import os

def preprocess_text(text):
    """Cleans and preprocesses text."""
    text = text.lower()
    text = re.sub('\s+', ' ', text).strip()
    return text

def tokenized_text(text):
    """Tokenizes text and returns tokenizer and word index."""
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])

    logging.info(f"Total word in vocab: {len(tokenizer.word_index)}")
    
    return tokenizer


def generate_padded_sequences(text, tokenizer):
    """Generates padded input sequences from tokenized text."""
    input_sequences = []
    
    for sentence in text.split("\n"):
        tokenized_sent = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(tokenized_sent)):
            input_sequences.append(tokenized_sent[:i+1])
            
    max_len = max([len(x) for x in input_sequences])
    
    padded_sequences = pad_sequences(input_sequences, maxlen=max_len, padding="pre")
    return padded_sequences, max_len


def main():
    """Processes the text step by step and returns training data."""
    try:
        with open("./data/raw/train.txt", "r") as file:
            text = file.read()
        logging.info("Preprocessing text...")
        text = preprocess_text(text)
        logging.info("Tokenizing text...")
        tokenizer = tokenized_text(text)
        logging.info("Generating padded sequences...")
        padded_sequences, max_len = generate_padded_sequences(text, tokenizer)

        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        # Save the padded sequences as a numpy array

        np.save(os.path.join(data_path, "padded_sequences.npy"), padded_sequences)
        
        # Optionally save tokenizer's word index as a dictionary in a text file
        with open(os.path.join(data_path, "word_index.txt"), "w") as f:
            for word, index in tokenizer.word_index.items():
                f.write(f"{word}: {index}\n")
        
        logging.info("Processing completed!")
        
    except Exception as e:
        logging.error('Failed to complete the data preprocessing: %s', e)
        print(f"Error: {e}")
        
        
if __name__=="__main__":
    main()
        


    
    
    
