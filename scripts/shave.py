import pandas as pd
import os

csvfile = "../trainLabels.csv"
directory = "../ML/Data/Kaggle-histopathology/train"

df = pd.read_csv(csvfile)
imagenames = [x.split(".")[0] for x in os.listdir("../ML/Data/Kaggle-histopathology/train")] 

for value in df["id"].values:
	df = df[value in imagenames]
	print(value)
	

# print(df["id"].values[0:5])
df.to_csv(r'../shaved.csv')

