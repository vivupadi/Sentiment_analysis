class vectorizer:
    def __init__(self):
        pass

    def bow_vect(self):


    def tf_idf_vect(self):


    def word2vec_vect(self):
        



def vectorize_data(df):
    stop_words = set(stopwords.words('english'))
    bow_counts = CountVectorizer(
    tokenizer=word_tokenize,
    stop_words=stop_words, #English Stopwords
    ngram_range=(1, 2) #analysis of one word
    )
    if 'Sentiment' in df.columns:
        df['Sentiment'] = df['Sentiment'].replace({'Positive': 1, 'Negative' : 0, 'Neutral': 2, 'Irrelevant': 3})
    text_vector = bow_counts.fit_transform(df['Text'])
    return text_vector, df['Sentiment'], bow_counts