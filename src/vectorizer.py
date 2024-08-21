import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import multiprocessing

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec
import gensim.downloader

#import fasttext

class vectorizer:
    def __init__(self):
        pass

    def bow_vect(self):
        stop_words = set(stopwords.words('english'))
        bow_counts = CountVectorizer(
        tokenizer=word_tokenize,
        stop_words=stop_words, #English Stopwords
        ngram_range=(1, 2) #analysis of one word
        )
        return bow_counts


    def tf_idf_vect(self):
        stop_words = set(stopwords.words('english'))
        return TfidfVectorizer(stop_words= stop_words, ngram_range=(1,4))


    def word2vec_vect(self):
        cores = multiprocessing.cpu_count()
        word2vec_model = Word2Vec(min_count=20,window=4,sample=6e-5, alpha=0.03, min_alpha=0.0007, 
                                  negative=20,workers=cores-1)
        return word2vec_model
    
    def glove_vect(self):
        glove_vectors = gensim.downloader.load('glove-twitter-25')
        return glove_vectors
    
    #def fasttext_vect(self):
        # Skipgram model :
     #   fasttext_model = fasttext.train_unsupervised('data.txt', model='skipgram')

        # or, cbow model :
      #  fasttext_model = fasttext.train_unsupervised('data.txt', model='cbow')
       # return fasttext_model