import numpy as np
import os
import math
from PIL import Image
import pandas as pd

# Clustering
from sklearn.mixture import GaussianMixture         # 3) Gaussian Mixture Models
from sklearn.preprocessing import StandardScaler

# Evaluation
from sklearn import metrics
from skimage.external import tifffile

from sklearn.decomposition import PCA               # 2) PCA

# Load Autoencoder                                  # 3) Autoencoder (Deep dimensionality reduction)
from keras.models import load_model
from keras.models import Model
import cv2
import argparse
import utilities as myutils

parser = argparse.ArgumentParser(description='Categorize this Whole Image Slide Into Tiles')
parser.add_argument('datapath', help='datapath for 1 WSI')
parser.add_argument('n_datatypes', help='estimated number of distinct types of tissue in this WSI')
parser.add_argument('magnification_level', help='options are 1.25, 2.5, 5.0, 10.0 or 20.0')

args=parser.parse_args()


if __name__ == '__main__':

	directory = args.datapath.split(".")[0] + "/" + (args.datapath.split(".")[0] + "_files") + "/" + args.magnification_level

	new_train = []
	image_names = []
	i = 0

	for filename in os.listdir(directory):
		if (filename.endswith('tif') or filename.endswith('jpeg') or filename.endswith('jpg') or filename.endswith('png')):
			image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_UNCHANGED)
			resized_image = cv2.resize(image, (96, 96)) 
			new_train.append(np.asarray( resized_image, dtype="uint8" ))
			image_names.append(filename)
			i+=1
			print("loading image " + filename + " number " + str(i))

	meanMatrix = np.asarray(new_train)

	autoencoder = load_model('../weights/AE_weight.h5') # 6 x 6 x 16
	layer_name = 'conv2d_7' # 6 x 6 x 16
	encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(layer_name).output)

	print("Reducing Dimensions")
	meanMatrix = meanMatrix.astype('float32') / 255. # Normalize the values before predictions
	X = encoder.predict(meanMatrix)

	X = X.reshape(X.shape[0] , -1) # Reshape for scaling
	X = StandardScaler().fit_transform(X) # Scale

	print("Reducing Dimensions again")
	pca = PCA(n_components=0.85)
	transformedData = pca.fit_transform(X)

	print("Clustering")
	gmm = GaussianMixture(n_components=int(args.n_datatypes), covariance_type='full', verbose=1).fit(transformedData)

	gmmlabels_ = gmm.predict(transformedData)
	
	myutils.clusterintoDirectories(gmmlabels_, directory, image_names)
	print("Finished clustering, you can find the symlinks now !")






