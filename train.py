import numpy as np
import pandas as pd
import os
import re
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer



def load_data(file_path):
    data = pd.read_csv(file_path, header = 0, names=["ID","Entity","Sentiment","Text"])
    #data = data[:100]
    return data

def preprocess(df):
    df['Text'] = df['Text'].str.lower()        # convert ot lowercase
    df['Text'] = [str(data) for data in df.Text]            #convert to string
    df['Text'] = df['Text'].apply(lambda x : re.sub('[^A-Za-z0-9 ]+', ' ', x)) #remove special characters
    df['Text'] = df['Text'].apply(lambda x : text_clean(x))
    return df

def text_clean(text):
    text = nltk.word_tokenize(text)
    stopwords_list = stopwords.words('english') #Remove stopword
    lemma = WordNetLemmatizer()
    words = [lemma.lemmatize(word) for word in text if word not in stopwords_list]
    summing = ' '.join(words)
    return summing


def vectorize_data(df):
    stop_words = set(stopwords.words('english'))
    bow_counts = CountVectorizer(
    tokenizer=word_tokenize,
    stop_words=stop_words, #English Stopwords
    ngram_range=(1, 1) #analysis of one word
    )
    df['Sentiment'] = df['Sentiment'].replace({'Positive': 1, 'Negative' : 0, 'Neutral': 2})
    text_vector = bow_counts.fit_transform(df['Text'])
    return text_vector, df['Sentiment']


#def load_model()

def train_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle =True)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy



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