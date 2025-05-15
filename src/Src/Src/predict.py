import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import clean_text

def predict_sentiment(text):
    with open("models/sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)

    tfidf = TfidfVectorizer()
    clean = clean_text(text)
    X = tfidf.fit_transform([clean])
    return model.predict(X)[0]
