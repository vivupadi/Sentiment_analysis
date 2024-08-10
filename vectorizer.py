import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer


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
        pass


    def word2vec_vect(self):
        pass