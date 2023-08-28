# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 21:05:36 2023

@author: Ramneek
"""

import streamlit as st
import pickle
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
#stopwords.words('english')
import string
string.punctuation
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()

def transform_text(text):
    text= text.lower()
    text= nltk.word_tokenize(text)
    y =[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

tfidf= pickle.load(open('C:/Users/Ramneek/OneDrive/Desktop/Email_SMS Classifier/vectorizer.pkl','rb'))
model= pickle.load(open('C:/Users/Ramneek/OneDrive/Desktop/Email_SMS Classifier/model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


    

