import pandas as pd
import html
from sklearn.metrics import cohen_kappa_score, accuracy_score
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

def test_train_split(X, y, test_size=0.2):
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=test_size, random_state=1337)
    return X_train, X_test, y_train, y_test

def agreement_accuracy_score(crowd, gold):
    df = crowd.merge(gold, on="text", suffixes=("_crowd", "_gold"))

    print("Aligned rows:", len(df))

    kappa = cohen_kappa_score(
        df["sentiment_crowd"],
        df["sentiment_gold"]
    )

    acc = accuracy_score(
        df["sentiment_crowd"],
        df["sentiment_gold"]
    )

    return (kappa,acc)



def feature_processing(cs):
        # replace twitter handles with @USER
    cs["text"] = cs["text"].str.replace(r"\B@\w+", "@USER", regex=True)

    # fix html unicode (&amp; -> &, &lt; -> <, etc.)
    cs["text"] = cs["text"].apply(html.unescape)

    # remove URLs
    cs["text"] = cs["text"].str.replace(r"http\S+|www\.\S+", "", regex=True)

def main():
    cs_train = pd.read_csv("data/crowdsourced_train.csv", sep="\t")
    gold_train = pd.read_csv("data/gold_train.csv", sep="\t")

    # normalize sentiment labels
    cs_train["sentiment"] = cs_train["sentiment"].str.lower().str.strip()
    gold_train["sentiment"] = gold_train["sentiment"].str.lower().str.strip()

    # keep only valid labels
    valid_labels = ["positive", "negative", "neutral"]
    cs_train = cs_train[cs_train["sentiment"].isin(valid_labels)]
    gold_train = gold_train[gold_train["sentiment"].isin(valid_labels)]

    feature_processing(cs_train)
    feature_processing(gold_train)
    
    agr_score, acc_score = agreement_accuracy_score(cs_train,gold_train)
    print("Agreement score (kappa) is: ",agr_score)
    print("Accuracy score is: ",acc_score)
    
    

if __name__ == "__main__":
    main()