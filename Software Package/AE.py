import numpy as np
import os
from sklearn.mixture import GaussianMixture         
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA               
from keras.models import load_model
from keras.models import Model
import cv2
import argparse
import utilities as myutils
from tqdm import tqdm
import sys

parser = argparse.ArgumentParser(description='Categorize this Whole Image Slide Into Tiles')
parser.add_argument('datapath', help='datapath for 1 WSI')
parser.add_argument('n_datatypes', help='estimated number of distinct types of tissue in this WSI')
parser.add_argument('magnification_level', help='options are 1.25, 2.5, 5.0, 10.0 or 20.0')
args=parser.parse_args()

if __name__ == '__main__':
	maindir = args.datapath.split(".")[0] + "/" + (args.datapath.split(".")[0] + "_files")
	new_train = []
	image_names = []
	magnifications = os.listdir(maindir)
	if '.DS_Store' in magnifications:
		magnifications.remove('.DS_Store')

	if args.magnification_level not in magnifications:
		closest_magnification = min(magnifications, key=lambda x:abs(float(x)- float(args.magnification_level)))
		print("Magnifications:")
		print(magnifications)
		print("Couldnt find the magnification {}, the closest one is {}".format(args.magnification_level, closest_magnification))
		magnification = str(closest_magnification)
	else:
		magnification = args.magnification_level

	directory = maindir + "/" + magnification
	if (len(os.listdir(directory)) == 0):
		print("magnification level {} directory doesnt have any tiles, please choose another magnification directory that has tiles".format(magnification))
		sys.exit(0)

	print("Loading images and preprocessing: ")
	for filename in tqdm(os.listdir(directory)):
		if (filename.endswith('tif') or filename.endswith('jpeg') or filename.endswith('jpg') or filename.endswith('png')):
			image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_UNCHANGED)
			resized_image = cv2.resize(image, (96, 96)) 
			new_train.append(np.asarray( resized_image, dtype="uint8" ))
			image_names.append(filename)

	meanMatrix = np.asarray(new_train)

	autoencoder = load_model('../weights/AE_weight.h5') # 6 x 6 x 16
	layer_name = 'conv2d_7' # 6 x 6 x 16
	encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(layer_name).output)

	print("Autoencoder compressing the images....")
	meanMatrix = meanMatrix.astype('float32') / 255. # Normalize the values before predictions
	X = encoder.predict(meanMatrix)

	X = X.reshape(X.shape[0] , -1) # Reshape for scaling
	X = StandardScaler().fit_transform(X) # Scale

	print("Reducing Dimensions....")
	pca = PCA(n_components=0.85)
	transformedData = pca.fit_transform(X)

	print("Clustering....")
	gmm = GaussianMixture(n_components=int(args.n_datatypes), covariance_type='full', verbose=1).fit(transformedData)

	gmmlabels_ = gmm.predict(transformedData)
	print("Clustering algorithm found {} classes, input was {}".format(len(set(gmmlabels_)), args.n_datatypes))

	if (len(set(gmmlabels_)) == 1):
		print("Only found 1 class, please try the other option")
	else:
		myutils.clusterintoDirectories(gmmlabels_, directory, image_names)
	
	print("Done")






