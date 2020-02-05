import pandas as pd
import os
from shutil import copy

mainDir = "C:/Users/mostafaosama2/Desktop/Kaggle-Data"
regressionDir = "C:/Users/mostafaosama2/Desktop/regressionSet"

# read labels into df
csvfile = mainDir + "/train_labels.csv"
df = pd.read_csv(csvfile)

# get the first 1000 train images
imgList = []
for file in os.listdir(mainDir + "/train")[0:80000]:
	if file.endswith('.tif'):
		imgList.append(file.split('.')[0])


for image in imgList:
	copy(os.path.join(mainDir + '/train', image + '.tif'), regressionDir + "/train")
	print(image)

imgtestList = []
for file in os.listdir(mainDir + "/test")[0:20000]:
	if file.endswith('.tif'):
		imgtestList.append(file.split('.')[0])
		
for image in imgtestList:
	copy(os.path.join(mainDir + '/test', image + '.tif'), regressionDir + "/test")
	print(image)

# adjust the df to contain the same images
labels_true = []
for image in imgList:
	label_index = df[df["id"] == image.split(".")[0]].index[0]
	labels_true.append(df["label"][label_index])


d = {'id': imgList, 'label': labels_true}
df_new = pd.DataFrame(data=d)

df = df[(df.id.isin(imgList))]

print(df_new)
df_new.to_csv(r'C:/Users/mostafaosama2/Desktop/regressionSet/first1000New.csv')

print(imgList == df_new["id"].values)