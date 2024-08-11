import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec
import gensim.downloader

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
        return TfidfVectorizer(max_features=5000)


    def word2vec_vect(self,sentence):
        word2vec_model = Word2Vec(sentences=sentence, vector_size=100, window=5, min_count=1, workers=4)
        return word2vec_model
    
    def glove_vect(self):
        glove_vectors = gensim.downloader.load('glove-twitter-25')
        return glove_vectors