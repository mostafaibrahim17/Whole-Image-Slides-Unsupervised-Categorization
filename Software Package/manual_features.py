import numpy as np
import os
import math
import shutil

from PIL import Image
import pandas as pd
# Clustering
from sklearn.mixture import GaussianMixture         # 3) Gaussian Mixture Models
from sklearn import metrics

from skimage.color import rgb2hed
from sklearn.preprocessing import StandardScaler
# Evaluation
from sklearn import metrics
from skimage.external import tifffile


import argparse
import utilities as myutils

parser = argparse.ArgumentParser(description='Categorize this Whole Image Slide Into Tiles')
parser.add_argument('datapath', help='datapath for 1 WSI')
parser.add_argument('n_datatypes', help='estimated number of distinct types of tissue in this WSI')
parser.add_argument('magnification_level', help='options are 1.25, 2.5, 5.0, 10.0 or 20.0')

args=parser.parse_args()


if __name__ == '__main__':

	directory = args.datapath.split(".")[0] + "/" + (args.datapath.split(".")[0] + "_files") + "/" + args.magnification_level
	i = 0
	mean_List = []
	image_names = []

	for filename in os.listdir(directory):
		if (filename.endswith('tif') or filename.endswith('jpeg') or filename.endswith('jpg') or filename.endswith('png')):
			i+=1
			image = Image.open(os.path.join(directory, filename))
			image_names.append(filename)
			mean_List.append(myutils.calculateMeanHE(image) + myutils.calculateMeanRGB(image))
			print("loading image " + filename + " number " + str(i))

	meanMatrix = np.asarray(mean_List)

	print("Scaling the data")
	# Standardize the dataset by removing the mean and scaling to unit variance z =(x-u)/s
	scaler = StandardScaler()
	scaledData = scaler.fit_transform(meanMatrix)

	print("Clustering " + args.n_datatypes + " clusters")
	gmm = GaussianMixture(n_components=int(args.n_datatypes), covariance_type='full', verbose=1).fit(scaledData)

	gmmlabels_ = gmm.predict(meanMatrix)
	print(gmmlabels_)
	
	myutils.clusterintoDirectories(gmmlabels_, directory, image_names)

	print("Done")









