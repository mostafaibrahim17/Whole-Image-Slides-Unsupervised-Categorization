import pandas as pd
import os

csvfile = "../ML/Data/Kaggle-original-copy/train_labels.csv"
directory = "../ML/Data/Kaggle-original-copy/train"

df = pd.read_csv(csvfile)

c = 1
for filename in os.listdir(directory):
	if not filename.startswith('.') and filename.endswith('tif'):
		os.rename(os.path.join(directory, filename), os.path.join(directory, 'Image {}'.format(c) + '.tif'))
		x = df["id"].replace({ filename.split(".")[0] :'Image {}'.format(c)})
		df.update(x)
		c += 1 
		print(f"Replaced entry, {filename} with entry Image {c}.tif")


df.to_csv(r'../NewKaggle.csv')

