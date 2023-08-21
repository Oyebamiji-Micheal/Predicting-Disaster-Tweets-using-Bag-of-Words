import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer

import streamlit as st
import pandas as pd
import joblib

st.write("## Intro to NLP: Predicting Disaster Tweets using Bag of Words")

st.write("""
A web app which predicts whether a given tweet is about a real disaster or
not using Bag of Words - A beginner's approach to NLP.
""")

st.image("images/cover.jpeg")

st.write("""
## About

<p align="justify">This project is focused on predicting whether a given tweet is related to a real disaster or not. The objective is to utilize NLP techniques to process and analyze the textual content of tweets and create a predictive model that can accurately classify tweets as disaster-related or not. The dataset was gotten from a Kaggle competition, <a href="https://www.kaggle.com/competitions/nlp-getting-started/" target="_blank">NLP Getting Started - Disaster Tweet Prediction</a>.
My main goal is to familiarize myself with fundamental NLP concepts, including tokenization, lemmatization, stop words, stemming, and the bag of words model. In addition, I aim to gain hands-on experience in applying NLP techniques to real-world text data and understanding their impact on model performance.</p>
""", unsafe_allow_html=True)

st.write("""
Everything you need to know regarding this project including the documentation, notebook, dataset can be found in my repository on [Github](https://github.com/Oyebamiji-Micheal/Predicting-Disaster-Tweets-using-Bag-of-Words).

Made by Oyebamiji Micheal
""")

st.sidebar.header("User Input")

tweet_input = st.sidebar.text_area("Enter the tweet:", "")

predict_tweet = st.sidebar.button("Predict Tweet")


def preprocess_tweet(tweet):
    # Remove links
    tweet = re.sub(r'http\S+', '', tweet)

    # Remove @username
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Tokenization
    tokens = word_tokenize(tweet)
    
    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalnum()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Perform stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return tokens


def predict_input(tweet_input):
    model_joblib = joblib.load('model.joblib')
    single_input = pd.DataFrame([{'text': tweet_input}])
    single_input = model_joblib['vectorizer'].transform(single_input.text)
    prediction = model_joblib['model'].predict(single_input)

    return prediction


if predict_tweet:
    prediction = predict_input(tweet_input)

    st.sidebar.write(f'Classifier = XGBoost')

    print(prediction)
    if prediction[0] == 1:
        st.sidebar.write('Predicted Status = Disaster Related Tweet')
    else:
        st.sidebar.write('Predicted Status = Not Disaster Related')
