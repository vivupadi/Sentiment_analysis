import numpy as np
import pandas as pd
import os
import re
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path, header = 0, names=["ID","Entity","Sentiment","Text"])
    return data

def preprocess(df):
    df['Text'] = df['Text'].str.lower()        # convert ot lowercase
    df['Text'] = [str(data) for data in df.Text]            #convert to string
    df['Text'] = df['Text'].apply(lambda x : re.sub('[^A-Za-z0-9 ]+', ' ', x))  #remove special characters
    df['Text'] = df['Text'].apply(lambda x : text_clean(x))
    return df

def text_clean(text):
    text = nltk.word_tokenize(text)
    stopwords_list = stopwords.words('english')
    lemma = WordNetLemmatizer()
    words = [lemma.lemmatize(word) for word in text if word not in stopwords_list]
    summing = ' '.join(words)
    return summing

file_path = "C:\\Users\\Vivupadi\\Desktop\\Sentiment Analysis\\data\\twitter_training\\twitter_training.csv"

data = load_data(file_path)
df = preprocess(data)
print(df)

"""
def preprocess(data):
    #only_aphabets
    #lowerupper_case
    #stop_words
    #tokenization
    #lemmatization
    #encodings
    return df_preprocessed

def train_model(df_preprocessed):
    #random frest
    #logreg
    #lstm
    return 


file_path = "C:\\Users\\Vivupadi\\Desktop\\Sentiment Analysis\\data\\twitter_training.csv"

"""