import pandas as pd
import os
from shutil import copy

mainDir = "../../Data/regrData/test"
csvfile = mainDir + "/train_labels.csv"
df = pd.read_csv(csvfile)
traindirectory = mainDir

for image in os.listdir(traindirectory):
	if image.endswith('.tif'):
		print(image)
		imagename = image.split('.')[0]
		label_index = df[df["id"] == imagename].index[0]
		label = df.iloc[[label_index]].values[0][1]
		if label == 0:
			copy(os.path.join(traindirectory, imagename + '.tif'), traindirectory+ "/0")
		elif label == 1:
			copy(os.path.join(traindirectory, imagename + '.tif'), traindirectory + "/1")



