# Sentiment Analysis
# 🧠 IMDB Sentiment Analysis App

A sleek and interactive Streamlit web app that uses a deep learning model to predict the sentiment of movie reviews. Enter a review, and the app tells you whether it's **Positive** or **Negative**, along with a confidence score and a progress bar.

This project uses a **Simple RNN (Recurrent Neural Network)** as the core of its sentiment analysis model.

A Simple RNN processes input sequences step-by-step, maintaining a hidden state that captures information from previous words. It's designed for sequence tasks like sentiment classification, where word order matters.


<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/21f6edce-a651-4420-b325-29be9bbf243f" />

---

## 🚀 Features

- 🔍 **Real-time sentiment prediction** from user input
- 🧠 **Deep learning model** built with TensorFlow/Keras
- 🌐 **Streamlit** web interface with intuitive UI
- 🎨 Aesthetic layout with color-coded results and emoji feedback
- 📊 Model confidence score and progress indicator

---

## 🗂️ Project Structure
├── app.py # Streamlit frontend app  
├── sentiment_analysis_model.h5 # Pre-trained sentiment classification model  
├── tokenizer.pickle (optional) # Tokenizer used during training (recommended)  
├── requirements.txt # Python dependencies   
└── README.md # This file  

### 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/imdb-sentiment-analyzer.git
   cd imdb-sentiment-analyzer
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Application**
   ```bash
   streamlit run app.py
   ```
