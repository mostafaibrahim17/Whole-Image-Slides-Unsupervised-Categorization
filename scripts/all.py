# Imports, mainly : numpy, pandas, sklearn, and matplotlib
import numpy as np
import os
from PIL import Image
import pandas as pd
from sklearn.cluster import KMeans
from skimage.color import rgb2hed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics

directory = "../new100"
mean_heList = []
PCAValuesList = []
image_names = []
labels_true = []
n_clusters = 2

# Feature extraction: Colors : meanHE, meanRGB, (max rgb and min rgb dont work)
def calculateMeanHE(image):
    hedImage = rgb2hed(image) # hed is short for Hematox, Eosin, DAB
    hemaValues = hedImage[:, :, 0]
    eosValues = hedImage[:, :, 1]
    return [hemaValues.mean(), eosValues.mean()]

def calculateMeanRGB(image):
    pixels = image.load()
    width, height = image.size 
    num_pixels = width * height
    r_mean = g_mean = b_mean = 0 
    for i in range(width):
        for j in range(height): 
            r,g,b=pixels[i,j] 
            r_mean += r
            g_mean += g
            b_mean += b
    return [r_mean/num_pixels , g_mean/num_pixels , b_mean/num_pixels]

# loop through each image in directory and calculate its values
for filename in os.listdir(directory):
    if not filename.startswith('.') and '.tif' in filename:
        image = Image.open(os.path.join(directory, filename))
        image_names.append(filename)
        mean_heList.append(calculateMeanHE(image))
        PCAValuesList.append(calculateMeanHE(image) + calculateMeanRGB(image))

# Fit values into K-Means
meanHEMatrix = np.asarray(PCAValuesList)
# Standardize the dataset by removing the mean and scaling to unit variance z =(x-u)/s
scaler = StandardScaler()
scaledData = scaler.fit_transform(meanHEMatrix)

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(scaledData)


# make cluster directories first
# then make a symbolic link of every image in the corresponding cluster directory
for cluster in range(n_clusters): 
	os.makedirs(directory + f'/{cluster}' , exist_ok=True)

# Check if there is a new file
sub_directories = [str(cluster) for cluster in range(n_clusters)]

# true labels is a list of 0s and 1s (cancerous & non-cancerous)

csvFilePath = "../first1000New.csv"
df = pd.read_csv(csvFilePath)

# Make sure the order of true labels and order of predicted labels match
for image in image_names:
    label_index = df[df["id"] == image.split(".")[0]].index[0]
    labels_true.append(df["label"][label_index])

evaluation = metrics.classification_report(labels_true, kmeans.labels_)
print(f'Kmeans: {evaluation}')
silhouette_avg = metrics.silhouette_score(meanHEMatrix, kmeans.labels_)
print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)

# PCA:

Matrix = np.asarray(PCAValuesList)
# Standardize the dataset by removing the mean and scaling to unit variance z =(x-u)/s
scaler = StandardScaler()
scaledData = scaler.fit_transform(Matrix)

pca = PCA(n_components=2)
transformedData = pca.fit_transform(scaledData)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(transformedData)

evaluation = metrics.classification_report(labels_true, kmeans.labels_)
print(f'PCA: {evaluation}')
silhouette_avg = metrics.silhouette_score(Matrix, kmeans.labels_)
print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)

for i in range(len(image_names)):
    # if there isnt already a symlink of this image in the coressponding subdirectory, make one
    if image_names[i] not in os.listdir(directory + '/' + sub_directories[kmeans.labels_[i]]): 
        os.symlink('../' + directory + f'/{image_names[i]}', 
                   directory + f'/{kmeans.labels_[i]}/{image_names[i]}')

############ Correction evaluation

# make a mapping between file names and labels for evaluation of the correction of symlinks
mappingdict = {}
for i in range(len(image_names)):
    mappingdict[image_names[i]] = kmeans.labels_[i]
    
# check kmeans labels against symbolic links (subdirectory name)
def evaluteSymbLinks():
	directoryLength = 0
	for subdirectory in sub_directories:
		directoryLength += len(os.listdir(directory + '/' + subdirectory))
		for image in os.listdir(directory + '/' + subdirectory):
			if str(mappingdict[image]) != subdirectory:
				return [False, directoryLength]
	return [True, directoryLength]

# check kmeans labels against symbolic links (subdirectory name)
# By checking that they are in the same order
def evaluteImagesCorrespondance():
	mylist = df["id"].values == [image.split(".")[0] for image in image_names]
	if False in mylist:
		return False
	return True 

# wont work with PCA
l = evaluteSymbLinks()
print(f'evaluted symbolic links to {l[0]}')
print(f'evaluted Images coresspondance to {evaluteImagesCorrespondance()}')
print(f'evaluted correct length {l[1]}')



















