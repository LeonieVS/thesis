import pandas as pd
import numpy as np
import re
import json
from pattern.nl import parse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import accuracy_score

path = "/Users/leonievanstappen/Documents/school/master/Thesis/news.csv"
df = pd.read_csv(path)
df.columns = ["naam", "titel", "tekst", "label"]
df = df.dropna()
y = df["label"]
df = df.drop(labels="label", axis=1)



X_train, X_test, y_train, y_test = train_test_split(df["tekst"], y, test_size=0.10)

#stijl tov inhoud: functiewoorden + punctuatie tov bag of words
#taalkundige informatie: dogwhistles
#niet te veel features
#get features
"""
def postags(articles):
    articles = X_train
    POS = {}
    for x in articles:
        POS[x] = parse(x, tags=True)
    return POS

#print(postags(X_train))
"""
#tfidf = TfidfVectorizer(max_df=1.0, max_features=500)
#tfidf_train = tfidf.fit_transform(X_train)
#tfidf_test = tfidf.transform(X_test)

#countvect = CountVectorizer(max_features=500)
#countvect_train = countvect.fit_transform(X_train)
#countvect_test = countvect.transform(X_test)

#clf = svm.SVC(kernel='linear')
#clf.fit(countvect_train, y_train)

#nb = MultinomialNB()
#nb.fit(countvect_train, y_train)

pipeline = Pipeline(steps=[("countvect", CountVectorizer()),
                            ("tfidf", TfidfTransformer()),
                            ("clf", svm.SVC(kernel='linear'))])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

print(accuracy_score(y_test, pred))
