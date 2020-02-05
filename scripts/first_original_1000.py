import pandas as pd
import os, shutil
from functools import reduce

csvfile = "../ML/Data/Kaggle-original/train_labels.csv"
csvfile1 = "../first1000.csv"

df = pd.read_csv(csvfile)

def evaluate():
	df = pd.read_csv(csvfile1)
	return ((df.id.values) == (imgList))

# dirlist = os.listdir("../ML/Data/Kaggle-original/train")[0:1960]
dirlist2 = os.listdir("../trydata")

# for image in dirlist:
# 	name = os.path.join("../ML/Data/Kaggle-original/train",image)
# 	shutil.copy(name, "../trydata")

# print(dirlist == dirlist2)

imgList = []

for x in dirlist2:
	if x.endswith('.tif'):
		imgList.append(x.split(".tif")[0])

# distinct = set(imgList)

# distinctdf = set(df.id.values)

# print((df.id.values) == (imgList))
# print((distinct.symmetric_difference(distinctdf)))
# print(len(distinct))
# print(len(distinctdf))

# df = reduce(pd.DataFrame.append, map(lambda i: df[df.id == i], imgList))
# df = df[(df.id.isin(imgList))]
print(evaluate())
# df.to_csv(r'../first1000.csv'))

