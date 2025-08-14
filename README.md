🎬 IMDB Movie Review Sentiment Analysis
This project is a Sentiment Analysis Web App that classifies IMDB movie reviews as Positive or Negative using a Simple RNN model built with TensorFlow/Keras.
The application is deployed using Streamlit for easy interaction.

📌 Features
🧠 Pre-trained Simple RNN Model (simple_rnn_imdb.h5)

📊 Sentiment classification of movie reviews

⌨️ User-friendly Streamlit interface

🔄 Text preprocessing & tokenization based on IMDB dataset word index

⚡ Real-time predictions

🎯 How It Works
IMDB Dataset Word Index is loaded from Keras.

Preprocessing:

Lowercasing

Tokenization using IMDB’s word index

Padding to a fixed sequence length (500 tokens)

The text is passed into the Simple RNN model.

Model outputs a probability score between 0 (negative) and 1 (positive).

Threshold: If score > 0.5 → Positive, else Negative.

📷 Example
Input:

css
Copy
Edit
This movie was absolutely amazing. The acting and story were top-notch!
Output:

makefile
Copy
Edit
Sentiment: Positive
Input:

kotlin
Copy
Edit
The plot was boring and predictable. I regret watching this.
Output:

makefile
Copy
Edit
Sentiment: Negative
📦 Requirements
From requirements.txt:

nginx
Copy
Edit
tensorflow
pandas
numpy
scikit-learn
tensorboard
matplotlib
streamlit
scikeras
📌 Notes
The provided model simple_rnn_imdb.h5 is trained on the IMDB dataset from Keras.

You can retrain the model using simplernn.ipynb or extend it with LSTM/GRU for improved performance.

This repo also contains other notebooks (prediction.ipynb, embedding.ipynb) for experimentation and learning.
