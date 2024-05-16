import subprocess
import sys

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install nltk if not already installed
try:
    import nltk
except ImportError:
    install('nltk')
    import nltk

# Now we can import the necessary modules from nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords and punkt if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

import streamlit as st
import pickle
import string

# Initialize PorterStemmer
ps = PorterStemmer()

# Function to transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
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

# Load TF-IDF vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app title and input
st.title('Email/SMS Classifier')
input_sms = st.text_input('Enter your message')
if st.button('Predict'):
    # Preprocessing
    transformed_sms = transform_text(input_sms)

    # Vectorization
    vector_input = tfidf.transform([transformed_sms])

    # Prediction
    result = model.predict(vector_input)[0]

    # Display result
    if result == 0:
        st.header('Spam')
    else:
        st.header('Not Spam')
