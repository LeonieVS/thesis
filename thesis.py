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
from sklearn.base import BaseEstimator, TransformerMixin #base classes to make it possible to work w feature union in pipeline
from sklearn import svm
from sklearn.metrics import accuracy_score



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
        for artikel in tekst:
            features["artikel"].append(artikel)
            #tok_artikel = nltk.word_tokenize(artikel)
            #features["artikel_pos"][i] = (" ").join([x[1] for x in nltk.pos_tag(tok_artikel)])#take the second element bc the first is the word repeated
        return features

if __name__ == "__main__":
    path = "/Users/leonievanstappen/Documents/school/master/Thesis/news.csv"
    df = pd.read_csv(path)
    df.columns = ["naam", "titel", "tekst", "label"]
    df = df.dropna()

    y = [1 if x == "neutral" else 0 for x in df["label"].values]
    X = df["tekst"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=44)

    #PIPELINE:
    feature_pipeline = Pipeline([
            ("features", FeatureExtractor()), #get text ready for analyzing: split into artikel text and pos tags
            ("union", FeatureUnion( #make one big feature matrix for all the different sets
            transformer_list=[

                #Pipeline for pulling features from articles

                ("punct_stats", Pipeline([
                    ("selector", ItemSelector(key="artikel")),
                    ("stats", Punct_Stats()),  # returns a list of dicts
                    ("vect", DictVectorizer()),  # list of dicts -> feature matrix
                ])),

                ("scheldwoorden", Pipeline([
                    ("selector", ItemSelector(key="artikel")),
                    ("stats", Profanity()),  # returns a list of dicts
                    ("vect", DictVectorizer()),  # list of dicts -> feature matrix
                ])),

                ("text_stats", Pipeline([
                    ("selector", ItemSelector(key="artikel")),
                    ("stats", Text_Stats()),  # returns a list of dicts
                    ("vect", DictVectorizer()),  # list of dicts -> feature matrix
                ])),

                ("counts", Pipeline([
                    ("selector", ItemSelector(key="artikel")),
                    ("vect", CountVectorizer()),  # list of dicts -> feature matrix
                ])),

            ],
        )),

        #use classifier on combined features
        ("classifier", MultinomialNB()) #change according to experiment/need --> I used MNNB to not get an error
    ])
    feature_pipeline.fit(X_train, y_train)

    scores = cross_val_score(feature_pipeline, X_test, y_test, cv=10)
    print(f"The average score in 10-fold cross-validation is {scores.mean()} with a standard deviation of {scores.std()}")
