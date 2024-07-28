from train import *

def predict_sentiment(model, vectorizer, text):
    if text:
        #df = pd.DataFrame()
        df = pd.DataFrame({'Text':[text]})
        df = preprocess(df)
        X = vectorize_predict(df, vectorizer)
        y = model.predict(X)
        sentiment = ['Negative', 'Positive', 'Neutral', 'Irrelevant']
        return sentiment[y[0]]

# same vectorizer need to be used. The purpose is to use the same vectorizer(Bag of Words) 
# to transform the text
def vectorize_predict(df, vectorizer):
    if 'Sentiment' in df.columns:
        df['Sentiment'] = df['Sentiment'].replace({'Positive': 1, 'Negative' : 0, 'Neutral': 2, 'Irrelevant': 3})
    #print(df['Sentiment'].unique())
    text_vector = vectorizer.transform(df['Text'])
    return text_vector