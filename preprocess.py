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

series = df.loc[df["naam"]=="Trouw", "tekst"]
regex = r"((Advertentie)((\n)+)(Teasers)((\n)+)\* )((Uitgelicht )(\|)((\n)+)\* (Net binnen)((\n)+)(Nam)?)"
series.replace(regex, "", inplace=True, regex=True)
df.loc[df["naam"]=="Trouw", "tekst"] = series
lengths = series.str.len()
#df.label = df.label.map({"rechts": -1, "neutral": 0, "links": 1})
print("Lengte:", len(df), "\nGemiddelde lengte:", lengths.mean(), "\nNiet neutraal:", len(df[df["label"]!=0]), "\nLinks:", len(df[df["label"]==1]), "\nRechts:", len(df[df["label"]==-1]), "\nNeutraal", len(df[df["label"]==0]))

#df.to_csv(path, index=False)
