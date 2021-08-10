import numpy as np
import os
from PIL import Image
from sklearn.mixture import GaussianMixture         # 3) Gaussian Mixture Models
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import argparse
import utilities as myutils

parser = argparse.ArgumentParser(description='Categorize this Whole Image Slide Into Tiles')
parser.add_argument('datapath', help='datapath for 1 WSI')
parser.add_argument('n_datatypes', help='estimated number of distinct types of tissue in this WSI')
parser.add_argument('magnification_level', help='options are 1.25, 2.5, 5.0, 10.0 or 20.0')
args=parser.parse_args()

if __name__ == '__main__':

	directory = args.datapath.split(".")[0] + "/" + (args.datapath.split(".")[0] + "_files") + "/" + args.magnification_level
	mean_List = []
	image_names = []

	print("Loading images and extracting features: ")
	for filename in tqdm(os.listdir(directory)):
		if (filename.endswith('tif') or filename.endswith('jpeg') or filename.endswith('jpg') or filename.endswith('png')):
			image = Image.open(os.path.join(directory, filename))
			image_names.append(filename)
			mean_List.append(myutils.calculateMeanHE(image) + myutils.calculateMeanRGB(image))

	meanMatrix = np.asarray(mean_List)

	print("Scaling the data")
	# Standardize the dataset by removing the mean and scaling to unit variance z =(x-u)/s
	scaler = StandardScaler()
	scaledData = scaler.fit_transform(meanMatrix)

	print("Clustering " + args.n_datatypes + " clusters")
	gmm = GaussianMixture(n_components=int(args.n_datatypes), covariance_type='full', verbose=1).fit(scaledData)

	gmmlabels_ = gmm.predict(meanMatrix)
	print("Clustering algorithm found {} classes, although input was {}".format(len(set(gmmlabels_)), args.n_datatypes))

	print("Finished clustering, copying to  sub-directories")

	if (len(set(gmmlabels_)) == 1):
		print("Only found 1 class, please try the other option")
	else:
		myutils.clusterintoDirectories(gmmlabels_, directory, image_names)

	print("Done")

