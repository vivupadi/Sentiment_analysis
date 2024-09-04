import numpy as np
import pandas as pd
import re

from models import *
from vectorizer import *

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


#nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV

def load_data(file_path):
    data = pd.read_csv(file_path, header = 0, names=["ID","Entity","Sentiment","Text"])
    data = data[:500]
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
    words = [lemma.lemmatize(word, pos='v') for word in text if word not in stopwords_list]
    summing = ' '.join(words)
    return summing


def vectorize_data(df, dropdown_vect):
    if 'Sentiment' in df.columns:
        df['Sentiment'] = df['Sentiment'].replace({'Positive': 1, 'Negative' : 0, 'Neutral': 2, 'Irrelevant': 3})
    selected_vect = vectorizer()
    if dropdown_vect == 'Bow':
        vect = selected_vect.bow_vect()
        text_vector = vect.fit_transform(df['Text'])
    
    elif dropdown_vect == 'TF_IDF':
        vect = selected_vect.tf_idf_vect()
        text_vector = vect.fit_transform(df['Text'])
    
    elif dropdown_vect == 'Word2vec':
        vect = selected_vect.word2vec_vect()
        vect.build_vocab(df['Text'])
        vect.train(df['Text'], total_examples=vect.corpus_count, epochs=10, report_delay=1)
        def get_sentence_vector(sentence, model):
            word_vectors = [model.wv[word] for word in sentence if word in model.wv.key_to_index]
            if word_vectors:
                return np.mean(word_vectors, axis=0)
            else:
                return np.zeros(model.vector_size)
        #Important concept in WOrd2Vec to convert the array
        text_vector = np.vstack(df['Text'].apply(lambda x: get_sentence_vector(x, vect)))
        print(text_vector)

    elif dropdown_vect == 'Glove':
        vect = selected_vect.glove_vect()
        text_vector = vect.fit(df['Text'])
    else:
        print('Select a vectorizer')

    return text_vector, df['Sentiment'], vect


def train_model(selected_model, X, y):
    mod = models()
    if selected_model == 'LogisticRegression':
        model, param = mod.log_reg()
    elif selected_model == 'XGBoost':
        model, param = mod.XGB()
    elif selected_model == 'Random Forest':
        model, param = mod.RanFo()
    else:
        print('Select model')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle =True)
    random_search = RandomizedSearchCV(model(), param_distributions=param, n_iter =6)
 
    random_search.fit(X_train, y_train)

    tuned_mod = model(**random_search.best_params_)

    tuned_mod.fit(X_train, y_train)

    y_pred = tuned_mod.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy = accuracy*100

    return tuned_mod, accuracy
