import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

path = "/Users/leonievanstappen/Documents/github/thesis/news.csv"
df = pd.read_csv(path)
df.columns = ["naam", "titel", "tekst", "label"]
df = df.dropna()
df = df.drop_duplicates()
deletdis = df[df["tekst"].map(len) < 500]
df = df.drop(deletdis.index)
df = df.reset_index(drop=True)
#print(len(df[df["label"]=="rechts"]))
series = df["tekst"]
lengths = series.str.len()
#for label in df.label.values:
#    print(type(label))
series2 = df.label
series2.replace("0", 0, inplace=True)
df.label = series2
#columns = ["tekst_pos", "titel_pos"]
#df = df.drop(columns, 1)
#df = df.drop(["tekst_pos", "titel_pos"], axis=1, inplace=True)
#df.label = df.label.map({"rechts": -1, "neutral": 0, "links": 1})
print("Lengte:", len(df), "\nGemiddelde lengte:", lengths.mean(), "\nNiet neutraal:", len(df[df["label"]!=0]), "\nLinks:", len(df[df["label"]==1]), "\nRechts:", len(df[df["label"]==-1]), "\nNeutraal", len(df[df["label"]==0]))
#print(df.label)
df.to_csv(path, index=False)
