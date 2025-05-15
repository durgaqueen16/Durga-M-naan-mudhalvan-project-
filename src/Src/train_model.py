import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from preprocess import clean_text

def train():
    df = pd.read_csv("data/social_media_data.csv")
    df['clean_text'] = df['text'].apply(clean_text)

    X = TfidfVectorizer().fit_transform(df['clean_text'])
    y = df['sentiment']  # assume column exists

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    with open("models/sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)
