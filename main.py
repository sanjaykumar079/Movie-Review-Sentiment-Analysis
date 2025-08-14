import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import traceback

# Verify model file exists
model_path = 'simple_rnn_imdb.keras'
if not os.path.exists(model_path):
    st.error(f"Model file {model_path} not found in {os.getcwd()}")
    raise FileNotFoundError(f"Model file {model_path} not found")

# Load word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load model with error handling
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {str(e)}\n\nFull traceback: {traceback.format_exc()}")
    raise

# Helper functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review')
if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    st.write(f'Sentiment: {sentiment}')
else:
    st.write('Please enter a movie review.')
