import pandas as pd
import numpy as np
import re
import json
from pattern.nl import parse

path = "/Users/leonievanstappen/Documents/school/master/Thesis/news.csv"
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
print(len(df), lengths.mean(), len(df[df["label"]=="links"]), len(df[df["label"]=="neutral"]), len(df[df["label"]=="rechts"]))

#deletdis = "Advertentie\n\nTeasers\n\n* Uitgelicht |\n* Net binnen\n\nNam"

#Advertentie\n\nTeasers\n\n* Uitgelicht |\n* Net binnen\n\nNam


#df.to_csv(path, index=False)
