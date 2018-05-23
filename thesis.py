from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion #make combination of heterogenous features possible
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score,recall_score
from spacy.tokenizer import Tokenizer
from spacy.pipeline import Tagger
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator, TransformerMixin #base classes to make it possible to work w feature union in pipeline
from sklearn.svm import SVC
from collections import defaultdict
from sys import argv
import pandas as pd
import numpy as np
import string #om de punctuation op te lijsten
import spacy
import nltk
import re


script, file =  argv


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, tekst):
        features = defaultdict(list)
        for titel, artikel, tekst_pos, titel_pos in tekst:
            features["artikel"].append(artikel)
            features["titel"].append(titel)
            features["tekst_pos"].append(tekst_pos)
            features["titel_pos"].append(titel_pos)
        return features


class Punct_Stats(BaseEstimator, TransformerMixin):
    #""""""Extract punctuation features from each document""""""

    def fit(self, x, y=None):
        return self

    def transform(self, tekst):
        punct_stats = []
        punctuations = list(string.punctuation)
        additional_punc = ["``", "--", "\"\""]
        punctuations.extend(additional_punc)
        for artikel in tekst:
            puncts = defaultdict(int)
            for ch in artikel:
                if ch in punctuations:
                    puncts[ch]+=1
            punct_stats.append(puncts)
        return punct_stats

class Profanity(BaseEstimator, TransformerMixin):
    #""""""Check artikeluments for ethnic slurs""""""

    def fit(self, x, y=None):
        return self

    def transform(self, tekst):
        scheldwoord = []
        slurs = open("scheldwoorden.txt", "r+")
        slurs = set(slurs.read().lower())
        for artikel in tekst:
            badwords = defaultdict(int)
            for word in artikel:
                if word in slurs:
                    badwords[word] += 1
            scheldwoord.append(badwords)
        return scheldwoord


class Text_Stats(BaseEstimator, TransformerMixin):
    #""""""Extract text statistics from each document""""""

    def fit(self, x, y=None):
        return self

    def transform(self, tekst):
        stats = []
        punctuation = string.punctuation
        with open("scheldwoorden.txt", "r+") as PROFANITY:
            for artikel in tekst:
                artikel_stats = {}
                tok_text = nltk.word_tokenize(artikel)
                abvs = ["EU", "US", "CNN", "BBC", "PS", "I", "USA", "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "GU", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MH", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "PR","PW", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VI", "VT", "WA", "WI", "WV", "WY", "UN", "BRB", "ASAP", "BTW", "VAT", "IDK", "CIA", "NAFTA", "USSR", "POTUS", "SCOTUS", "FLOTUS", "CPS", "DNC", "RNC", "NASA", "FBI", "AWOL", "MIA", "POW", "CSI", "ADD", "ADHD", "HIV", "CDC", "EST", "PST", "PC", "TV", "CST", "SSN", "SUV", "FYI", "UFO", "DOB", "AKA", "ETA", "IMO", "NFL", "AARP", "AA", "APA", "MLA", "SPCA", "PAWS", "PBS", "CBS", "ESPN", "ABC", "MTV", "ATM", "AAA", "WWE", "ACS", "NRA", "TNT", "AMA", "ADA", "DACA", "ACLU"]
                try: #all caps
                    num_upper = float(len([w for w in tok_text if w.isupper() and w not in abvs]))/len(tok_text)
                except ZeroDivisionError:
                    num_upper = 0
                try: #how much of the text is punctuation
                    num_punct = float(len([ch for ch in artikel if ch in punctuation]))/len(artikel)
                except ZeroDivisionError:
                    num_punct = 0
                try: #how much of text is profanity
                    num_prof = float(len([w for w in tok_text if w.lower() in PROFANITY]))/len(tok_text)
                except ZeroDivisionError:
                    num_prof = 0
                try: #how long are the
                    sent_lengths = [len(nltk.word_tokenize(s)) for s in nltk.sent_tokenize(artikel)]
                    av_sent_len = float(sum(sent_lengths))/len(sent_lengths)
                except ZeroDivisionError:
                    av_sent_len = 0

                artikel_stats["all_caps"] = num_upper
                artikel_stats["punctuation"] = num_punct
                artikel_stats["profanity"] = num_prof
                artikel_stats["sent_len"] = av_sent_len
                stats.append(artikel_stats)
        return stats


def PipelineCreator():
    pipeline = Pipeline([
            ("features", FeatureExtractor()),
            ("union", FeatureUnion( #make one big feature matrix from all the different sets
            transformer_list=[

                #Pipeline for pulling features from articles
            ("punct_stats_body", Pipeline([
                          ("selector", ItemSelector(key="artikel")),
                          ("stats", Punct_Stats()),  # returns a list of dicts
                          ("vect", DictVectorizer()),  # list of dicts -> feature matrix
                      ])),

            ("scheldwoorden_body", Pipeline([
                      ("selector", ItemSelector(key="artikel")),
                      ("stats", Profanity()),  # returns a list of dicts
                      ("vect", DictVectorizer()),  # list of dicts -> feature matrix
                  ])),

            ("text_stats_body", Pipeline([
                      ("selector", ItemSelector(key="artikel")),
                      ("stats", Text_Stats()),  # returns a list of dicts
                      ("vect", DictVectorizer()),  # list of dicts -> feature matrix
                  ])),

            ("counts_body", Pipeline([
                      ("selector", ItemSelector(key="artikel")),
                      ("vect", CountVectorizer(ngram_range=(1,3), token_pattern = r"\b\w+\b", max_df = 0.5)),  # list of dicts -> feature matrix
                  ])),

            ("ngrams_body", Pipeline([
                      ("selector", ItemSelector(key="artikel")),
                      ("vect", TfidfVectorizer(ngram_range=(1,3), token_pattern = r"\b\w+\b", max_df = 0.5)),
                 ])),

            ("POS_body", Pipeline([
                     ("selector", ItemSelector(key="tekst_pos")),
                     ("vect", CountVectorizer(ngram_range=(1,3), token_pattern = r"\b\w+\b", max_df = 0.5)),
                ])),

            ("punct_stats_titel", Pipeline([
                             ("selector", ItemSelector(key="titel")),
                             ("stats", Punct_Stats()),  # returns a list of dicts
                             ("vect", DictVectorizer()),  # list of dicts -> feature matrix
                         ])),

            ("scheldwoorden_titel", Pipeline([
                     ("selector", ItemSelector(key="titel")),
                     ("stats", Profanity()),  # returns a list of dicts
                     ("vect", DictVectorizer()),  # list of dicts -> feature matrix
                 ])),

            ("text_stats_titel", Pipeline([
                     ("selector", ItemSelector(key="titel")),
                     ("stats", Text_Stats()),  # returns a list of dicts
                     ("vect", DictVectorizer()),  # list of dicts -> feature matrix
                 ])),

            ("counts_titel", Pipeline([
                     ("selector", ItemSelector(key="titel")),
                     ("vect", CountVectorizer(ngram_range=(1,3), token_pattern = r"\b\w+\b", max_df = 0.5)),  # list of dicts -> feature matrix
                 ])),

            ("ngrams_titel", Pipeline([
                      ("selector", ItemSelector(key="titel")),
                      ("vect", TfidfVectorizer(ngram_range=(1,3), token_pattern = r"\b\w+\b", max_df = 0.5)),
                 ])),

            ("POS_titel", Pipeline([
                      ("selector", ItemSelector(key="titel_pos")),
                      ("vect", CountVectorizer(ngram_range=(1,3), token_pattern = r"\b\w+\b", max_df = 0.5)),
                 ])),

            ]

        )),
        ("selection", SelectKBest(k=1500, score_func=f_classif)),
        ("classifier", RandomForestClassifier(n_estimators=20, criterion="entropy"))
    ])

    return pipeline


def trainmodel(model):
    #train model for neutral/non-neutral classification
    df = model
    df.columns = ["naam", "titel", "tekst", "label", "tekst_pos", "titel_pos", "headlinebody"]
    df["headlinebody"] = list(zip(df["titel"], df["tekst"], df["tekst_pos"], df["titel_pos"]))
    y = df.label.values
    X = df.headlinebody.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=30)
    y_neut = [0 if x == 0 else 1 for x in y_train]
    neutrality_pipeline = PipelineCreator()
    neutrality_fitted = neutrality_pipeline.fit(X_train, y_neut)

    #train model for left-right bias classification
    indices = []
    for idx, label in enumerate(y_train):
        if label == 0:
            indices.append(idx)
    yBias = np.delete(y_train, indices)
    XBias = np.delete(X_train, indices)
    bias_pipeline = PipelineCreator()

    bias_fitted = bias_pipeline.fit(XBias, yBias)
    return bias_fitted, neutrality_fitted


def classify(testdata, pipeline_1, pipeline_2):
#classify the data in test/development set in two steps
    responses = []
    prediction = pipeline_1.predict(testdata)
    for i, line in enumerate(testdata):
        if prediction[i] == 0:
            result = 0
        else:
            scores = pipeline_2.predict_proba([testdata[i]])[0]
            if scores[1]>scores[0]:
                result = 1
                #result = scores[1]
            elif scores[0]>scores[1]:
                result = -1
                #result = scores[0]*-1
            else:
                result = 0
        responses.append(result)
    return responses

if __name__ == "__main__":
    with open(file,"r+",) as open_csv:
        df = pd.read_csv(open_csv, keep_default_na=False, index_col=False)
        df.columns = ["naam", "titel", "tekst", "label", "tekst_pos", "titel_pos"]
    df["headlinebody"] = list(zip(df["titel"], df["tekst"], df["tekst_pos"], df["titel_pos"]))

    y = df.label.values
    X = df.headlinebody.values

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    #accuracy = []
    prec = []
    rec = []
    f1 = []
    iter = 1
    for train, test in kfold.split(X, y):
        print(iter)
        X_train = df.headlinebody.values[train]
        y_train = df.label.values[train]
        X_test = df.headlinebody.values[test]
        y_test = df.label.values[test]

        y_neut = [0 if x == 0 else 1 for x in y_train]
        indices = []
        for idx, label in enumerate(y_test):
            if label == 0:
                indices.append(idx)
        yBias = np.delete(y_test, indices)
        XBias = np.delete(X_test, indices)

        bias_fitted, neutrality_fitted = trainmodel(df)
        responses = classify(X_test, neutrality_fitted, bias_fitted)

        precision = precision_score(y_test, responses, labels=[-1, 0, 1], average="macro")
        recall = recall_score(y_test, responses, labels=[-1, 0, 1], average="macro")
        f1score = f1_score(y_test, responses, labels=[-1, 0, 1], average="macro")

        print("Confusion Matrix:\n", confusion_matrix(y_test, responses, labels=[-1, 0, 1]))
        #print("accuracy:\n", accuracy_score(y_test, responses))
        print(classification_report(y_test, responses))

        prec.append(precision)
        rec.append(recall)
        f1.append(f1score)
        iter += 1
    print("\nP:", np.asarray(prec).mean(), "\nP std:", np.asarray(prec).std(), "\nR:", np.asarray(rec).mean(), "\nR std:", np.asarray(rec).std(), "\nF1:", np.asarray(f1).mean(), "\nF1 std:", np.asarray(f1).std())



    """#GridSearch: find best parameters for each classifier and for KBest selection using train set to avoid overfitting
    kbest_param = [{"selection__k": [10, 50, 100, 500, 1000, 1500], "selection__score_func": [f_classif, chi2]}]
    rfc_parameters = [{"classifier__n_estimators": [10, 20], "classifier__criterion": ["gini", "entropy"]}]
    SVM_parameters = [{"classifier__C": [1, 5], "classifier__kernel": ["rbf", "sigmoid"], "classifier__gamma": [0.01, 0.001], "classifier__shrinking": [True, False]}] #test again with poly kernel and degrees
    knn_parameters = [{"classifier__n_neighbors": [5, 15], "classifier__weights": ["uniform", "distance"], "classifier__p": [1, 2]}]
    nb_parameters = [{"classifier__alpha": [0.0, 1.0], "classifier__fit_prior": [True, False], "classifier__p": [1, 2]}]

    pipeline = PipelineCreator(classifier, selection)
    scoring = ["accuracy"]
    for score in scoring:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(pipeline, nb_parameters, cv=5, scoring="%s" % score, n_jobs=10)
        clf.fit(X_train, y_train)
        print("Best parameters set found on train set:")
        print(clf.best_params_)
        print()
        #the following code shows all possible parameter combinations and their scores

        print("Grid scores on train set:")
        print()
        means = clf.cv_results_["mean_test_score"]
        stds = clf.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))"""
