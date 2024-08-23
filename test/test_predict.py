from src.predict import predict_sentiment

def test_predict_sentiment():
    model = load_model('models/sentiment_model.pkl')
    text = "I love Nvidia"
    result = predict_sentiment(model, text)
    assert result == 'Positive'