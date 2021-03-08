import numpy as np
import os
import shutil

import math
from PIL import Image
import pandas as pd

# Clustering
from sklearn.cluster import AgglomerativeClustering # 1) Agglomerative-Hierarchical
from sklearn.cluster import KMeans                  # 2) K-Means
from sklearn.mixture import GaussianMixture         # 3) Gaussian Mixture Models

from skimage.color import rgb2hed
from sklearn.preprocessing import StandardScaler

# Evaluation
from sklearn import metrics
from skimage.external import tifffile
from skimage.measure import compare_mse
# from skimage.metrics import structural_similarity as ssim

from sklearn.decomposition import PCA, KernelPCA               # 2) PCA
from sklearn.manifold import TSNE

# Load Autoencoder                                  # 3) Autoencoder (Deep dimensionality reduction)
from keras.models import load_model
from keras.models import Model
from tqdm import tqdm
import cv2

import matplotlib.pyplot as plt 

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

# This might not work for windows
def symlink_rel(src, dst):
    rel_path_src = os.path.relpath(src, os.path.dirname(dst))
    os.symlink(rel_path_src, dst)

def clusterintoDirectories(labels, path, imagenamesList):
    # remove existing subdirectories first to avoid overlap
    sub_directories = [str(cluster) for cluster in set(labels)]
    # print(sub_directories)

    for cluster in tqdm(sub_directories):
        if (cluster in os.listdir(path)) and (os.path.isdir(os.path.join(path , cluster))):
            shutil.rmtree(os.path.join(path , cluster))

    print("Making sub-directories:")
    for filename in tqdm(os.listdir(path)):
        if (filename.endswith('tif') or filename.endswith('jpeg') or filename.endswith('jpg') or filename.endswith('png')):
            for cluster in sub_directories: # count of distinct elements = no. of clusters
                # print("Making subdirectories")
                os.makedirs(path + '/{}'.format(cluster) , exist_ok=True)
    
    print("Moving clustered tiles to sub-directories:")
    for i in tqdm(range(len(imagenamesList))):
        # if there isnt already a symlink of this image in the coressponding subdirectory
        if imagenamesList[i] not in os.listdir(path + '/' + sub_directories[sub_directories.index(str(labels[i]))]): 
            symlink_rel(path + '/{}'.format(imagenamesList[i]) , 
                       path + '/{}'.format(labels[i]) + '/' + imagenamesList[i])

