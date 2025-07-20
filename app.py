import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# load the IMDB dataset word index
word_index = tf.keras.datasets.imdb.get_word_index()
# reverse the word index to get words from indices
reverse_word_index = {value: key for key, value in word_index.items()}

# load the pre-trained model
model = tf.keras.models.load_model('sentiment_analysis_model.h5')

# function to decode the review from indices to words
def decode_review(text):
    """Decode the review from indices to words."""
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text]) 

# function to preprocess the input text
def preprocess_text(text, vocab_size=10000):
    """Preprocess the input text for prediction."""
    words = text.lower().split()
    # encoded review
    encoded = [min(word_index.get(word, 2) + 3, vocab_size - 1) for word in words]
    # pad the sequences
    padded = pad_sequences([encoded], maxlen=500)
    return padded

# function to predict sentiment
def predict_sentiment(text):
    """Predict the sentiment of the input text."""
    padded_text = preprocess_text(text)
    prediction = model.predict(padded_text)
    return prediction[0][0]

# Streamlit app
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

# App title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 IMDB Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a movie review to determine whether the sentiment is positive or negative.</p>", unsafe_allow_html=True)

# Text input
text_input = st.text_area("💬 Your Review", placeholder="Type your movie review here...", height=200)

# Analyze button
if st.button("Analyze Sentiment"):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        prediction_score = predict_sentiment(text_input)
        
        if prediction_score > 0.6:
            sentiment = "😊 Positive"
            color = "green"
        elif prediction_score < 0.4:
            sentiment = "😠 Negative"
            color = "red"
        else:
            sentiment = "😐 Neutral"
            color = "orange"

        st.markdown(f"<h2 style='text-align: center; color:{color};'>Sentiment: {sentiment}</h2>", unsafe_allow_html=True)
        st.progress(float(prediction_score))
        st.caption(f"Model confidence: {prediction_score:.2f}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; font-size: 0.9em;'>Built with ❤️ using Streamlit and TensorFlow</p>", unsafe_allow_html=True)