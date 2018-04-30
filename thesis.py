import pandas as pd
import numpy as np
import re
import nltk
import string #om de punctuation op te lijsten
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion #make combination of heterogenous features possible
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin #base classes to make it possible to work w feature union in pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix



class ItemSelector(BaseEstimator, TransformerMixin): #picks data from dict(X) according to key, returns one-dimensional array (vector)
    def __init__(self, key): #establish keyword arguments
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict): #itemselector selects from given dict according to certain key
        return data_dict[self.key]

class Punct_Stats(BaseEstimator, TransformerMixin):#BaseEstimator and TransformerMixin make it possible to transform data according to defined functions, returns np array
    #""""""Extract punctuation features from each document""""""

    def fit(self, x, y=None):
        return self

    def transform(self, tekst): #transform fct makes it so it's usable in a pipeline
        punct_stats = []
        punctuations = list(string.punctuation)
        additional_punc = ["``", "--", "\"\""]
        punctuations.extend(additional_punc)
        for artikel in tekst:
            puncts = defaultdict(int) #manier om dict te initializeren zodat als een key niet bekend is er een nieuwe wordt aangemaakt met bep integer (hier 0)
            for ch in artikel:
                if ch in punctuations:
                    puncts[ch]+=1
            punct_stats.append(puncts)
        return punct_stats #kan sklearn gemakkelijk features van maken --> dictvectorizer

class Profanity(BaseEstimator, TransformerMixin):
    #""""""Check artikeluments for ethnic slurs""""""

    def fit(self, x, y=None):
        return self

    def transform(self, tekst):
        scheldwoord = []
        slurs = open("scheldwoorden.txt", "r+")
        slurs = set(slurs.read().lower())
        for artikel in tekst:
            badwords = defaultdict(int) #way to initialize dict so that if key is unknown, new key is made with value: integer (0 here)
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
                try:
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


class FeatureExtractor(BaseEstimator, TransformerMixin): #takes care of POS --> uit nltk
    #Extracts PoS-tags from each document
    def fit(self, x, y=None):
        return self

    def transform(self, tekst):
        features = defaultdict(list)
        for titel, artikel in tekst:
            features["artikel"].append(artikel)
            features["titel"].append(titel)
            #tok_artikel = nltk.word_tokenize(artikel)
            #features["artikel_pos"][i] = (" ").join([x[1] for x in nltk.pos_tag(tok_artikel)])#take the second element bc the first is the word repeated
        return features


def PipelineCreator():
    pipeline = Pipeline([
            ("features", FeatureExtractor()), #get text ready for analyzing: split into artikel text and pos tags
            ("union", FeatureUnion( #make one big feature matrix for all the different sets
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
                    ("vect", CountVectorizer(ngram_range=(1,3), token_pattern = r'\b\w+\b', max_df = 0.5)),  # list of dicts -> feature matrix
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
                    ("vect", CountVectorizer(ngram_range=(1,3), token_pattern = r'\b\w+\b', max_df = 0.5)),  # list of dicts -> feature matrix
                ])),

                ('ngrams_titel', Pipeline([
                     ('selector', ItemSelector(key='titel')),
                     ('vect', TfidfVectorizer(ngram_range=(1,3), token_pattern = r'\b\w+\b', max_df = 0.5)),
                ])),

                 ('ngrams_body', Pipeline([
                     ('selector', ItemSelector(key='artikel')),
                     ('vect', TfidfVectorizer(ngram_range=(1,3), token_pattern = r'\b\w+\b', max_df = 0.5)),
                ])),
            ],
        )),

        #use classifier on combined features
        ("classifier", LogisticRegression(penalty="l2"))
    ])

    return pipeline


def trainmodel(path):
    #train model for neutral/non-neutral classification
    df = pd.read_csv(path)
    df.columns = ["naam", "titel", "tekst", "label"]
    df["headlinebody"] = list(zip(df["titel"], df["tekst"]))
    y = [1 if x == 0 else 0 for x in df["label"].values]
    X = df.headlinebody.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=44)
    neutrality_pipeline = PipelineCreator()
    neutrality_fitted = neutrality_pipeline.fit(X_train, y_train)
    #train model for left-right bias classification
    indices = []
    for idx, label in df.label.iteritems():
        if label == 0: #drop all neutral labels
            indices.append(idx)

    y2 = df.label.drop(indices)#only non-neutral labels remain
    X2 = df.headlinebody.drop(indices).values#drop neutral rows from data
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.10, random_state=44)
    bias_pipeline = PipelineCreator()
    bias_fitted = bias_pipeline.fit(X2_train, y2_train)
    print(len(X_train), len(X2_train))
    return neutrality_fitted, bias_fitted

def classify(testdata, pipeline_1, pipeline_2):
    """ Classifies inputs """
    responses = []
    prediction = pipeline_1.predict(testdata)

    for i, line in enumerate(testdata): #index and articles text for X_test
        if prediction[i] == 0:
            result = 0
        else:
            scores = pipeline_2.predict_proba([testdata[i]])[0]
            if scores[1]>scores[0]:
                result = scores[1]
            elif scores[0]>scores[1]:
                result = scores[0]*-1
            else:
                result = 0
        line = (line, result)
        responses.append(line)
    return responses


if __name__ == "__main__":
    path = "/Users/leonievanstappen/Documents/github/thesis/news.csv"
    df = pd.read_csv(path)
    df.columns = ["naam", "titel", "tekst", "label"]
    df["headlinebody"] = list(zip(df["titel"], df["tekst"]))

    y = [1 if x == 0 else 0 for x in df["label"].values]
    X = df["headlinebody"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=44)

    bias_fitted, neutrality_fitted = trainmodel(path)

    scores = classify(X_test, neutrality_fitted, bias_fitted)

    for x in scores:
        print(x[-1])

    neut_scores = cross_val_score(neutrality_fitted, X_test, y_test, cv=10)
    print(f"The average score in 10-fold cross-validation is {neut_scores.mean()} with a standard deviation of {neut_scores.std()}")
    #print("This is the confusion matrix:\n", confusion_matrix(y_test, pred))

    #sklearn precision recall confusion matrix
