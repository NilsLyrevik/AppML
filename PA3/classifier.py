import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



# FEATURE REPRESENTATION USING TF-IDF
# Skriv lite mer om vad TF-IDF är HÄR
def tfidf_vectorize(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X

# CLASSIFIER 1: Naive Bayes
# Skriv lite mer om Naive Bayes HÄR
def train_naive_bayes(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# CLASSIFIER 2: Logistic Regression
# Skriv lite mer om Logistic Regression HÄR
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# helpers for reading csv files (all have two columns: 'sentiment' and 'text')
def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df['sentiment'], df['label']

def test_train_split(X, y, test_size=0.2):
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=test_size, random_state=1337)
    return X_train, X_test, y_train, y_test