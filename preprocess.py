import pandas as pd

path = "/Users/leonievanstappen/Documents/school/master/Thesis/news.csv"
df = pd.read_csv(path)
df.columns = ["naam", "titel", "tekst", "label"]
df = df.dropna()
df = df.drop_duplicates()
print(len(df))
df.to_csv(path, index=False)
