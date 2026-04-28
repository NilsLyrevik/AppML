# Imports
import pandas as pd
import html
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# FEATURE REPRESENTATION 
# TFIDF vectorizer
def tfidf_vectorize(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X
# BoW vectorizer
def bow_vectorize(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X
# Word embeddings vectorizer
def word_embeddings_vectorize(corpus, embedding_model):
    # This is a placeholder function. In practice, you would load a pre-trained embedding model (like GloVe or Word2Vec)
    # and average the word vectors for each document in the corpus.
    pass

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

# 
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

# helper function to plot kappa and accuracy scores for all models (to remove bloat from main)
def plot_model_kappa_accuracy(kappa_scores, acc_scores):
    models = ["Majority Class", "Naive Bayes", "Logistic Regression"]
    x = range(len(models))
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(x, kappa_scores, color=['blue', 'orange', 'green'])
    plt.xticks(x, models)
    plt.title('Kappa Scores by Model')
    plt.ylabel('Kappa Score')
    plt.subplot(1, 2, 2)
    plt.bar(x, acc_scores, color=['blue', 'orange', 'green'])
    plt.xticks(x, models)
    plt.title('Accuracy Scores by Model')
    plt.ylabel('Accuracy Score')
    plt.tight_layout()
    # save plot as image 
    plt.savefig("plots/model_kappa_accuracy.png")


def feature_processing(cs):
        # replace twitter handles with @USER
    cs["text"] = cs["text"].str.replace(r"\B@\w+", "@USER", regex=True)

    # fix html unicode (&amp; -> &, &lt; -> <, etc.)
    cs["text"] = cs["text"].apply(html.unescape)

    # remove URLs
    cs["text"] = cs["text"].str.replace(r"http\S+|www\.\S+", "", regex=True)

# THIS IS WHERE ENTIRE PIPELINE HAPPENS!
def main():
    # read/ load data
    cs_train = pd.read_csv("data/crowdsourced_train.csv", sep="\t")
    gold_train = pd.read_csv("data/gold_train.csv", sep="\t")

    # normalize sentiment labels
    cs_train["sentiment"] = cs_train["sentiment"].str.lower().str.strip()
    gold_train["sentiment"] = gold_train["sentiment"].str.lower().str.strip()

    # keep only valid labels
    valid_labels = ["positive", "negative", "neutral"]
    cs_train = cs_train[cs_train["sentiment"].isin(valid_labels)]
    gold_train = gold_train[gold_train["sentiment"].isin(valid_labels)]

    # preprocessing
    feature_processing(cs_train)
    feature_processing(gold_train)
    
    # scores for writing later
    agr_score, acc_score = agreement_accuracy_score(cs_train,gold_train)
    print("Agreement score (kappa) is (when comparing datasets): ",agr_score)
    print("Accuracy score is (when comparing datasets): ",acc_score)

    # Use-TF-IDF vectorizer
    X_cs = tfidf_vectorize(cs_train["text"])
    y_cs = cs_train["sentiment"]
    X_gold = tfidf_vectorize(gold_train["text"])
    y_gold = gold_train["sentiment"]

    # Majority class model (always predict the most common class in the training data (neutral in this case))
    # This is our trivial baseline
    majority_class = y_cs.mode()[0]
    majority_predictions = [majority_class] * len(y_gold)
    majority_kappa = cohen_kappa_score(majority_predictions, y_gold)
    majority_acc = accuracy_score(majority_predictions, y_gold)
    print("Majority class Kappa :", majority_kappa)
    print("Majority class Accuracy :", majority_acc)
    
    # Naive Bayes Model, Model nr 1
    X_train_nb, X_test_nb, y_train_nb, y_test_nb = test_train_split(X_gold, y_gold)
    nb_model = train_naive_bayes(X_train_nb, y_train_nb)
    nb_predictions = nb_model.predict(X_test_nb)
    nb_kappa = cohen_kappa_score(nb_predictions, y_test_nb)
    nb_acc = accuracy_score(nb_predictions, y_test_nb)
    print("Naive Bayes Kappa :", nb_kappa)
    print("Naive Bayes Accuracy :", nb_acc)    

    # Logistic Regression Model, Model nr 2 
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = test_train_split(X_gold, y_gold)
    lr_model = train_logistic_regression(X_train_lr, y_train_lr)
    lr_predictions = lr_model.predict(X_test_lr)
    lr_kappa = cohen_kappa_score(lr_predictions, y_test_lr)
    lr_acc = accuracy_score(lr_predictions, y_test_lr)
    print("Logistic Regression Kappa :", lr_kappa)
    print("Logistic Regression Accuracy :", lr_acc)

    # plot all models kappa and accuracy scores using helper
    kappa_scores = [majority_kappa, nb_kappa, lr_kappa]
    acc_scores = [majority_acc, nb_acc, lr_acc]
    plot_model_kappa_accuracy(kappa_scores, acc_scores)



    

if __name__ == "__main__":
    main()