import numpy as np
import os
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
from skimage.metrics import structural_similarity as ssim

from sklearn.decomposition import PCA, KernelPCA               # 2) PCA
from sklearn.manifold import TSNE

# Load Autoencoder                                  # 3) Autoencoder (Deep dimensionality reduction)
from keras.models import load_model
from keras.models import Model

import cv2

import matplotlib.pyplot as plt 

def loadData(datapath): # returns images nparray and ImagesNames
    returnList = []
    mean_List = []
    image_names = []

    for filename in os.listdir(datapath):
        if filename.endswith('tif') :
            image = Image.open(os.path.join(datapath, filename))
            image_names.append(filename)
            mean_List.append(calculateMeanHE(image) + calculateMeanRGB(image))

    meanMatrix = np.asarray(mean_List)

    returnList.append(mean_List)
    returnList.append(image_names)
    return returnList

def loadLabelsFromcsv(csvFilePath):
    df = pd.read_csv(csvFilePath)
    testlabels = []

    for image in image_names:
        label_index = df[df["id"] == image.split(".")[0]].index[0]
        testlabels.append(df["label"][label_index])

    return testlabels

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

def ClusterAndPlot(n_clusters, D):
    Labels = []
    
    print(D.shape)
    HC = AgglomerativeClustering(n_clusters=n_clusters, affinity='manhattan', linkage='complete').fit(D)
    print('HC Silhouette Score  {} '.format(metrics.silhouette_score(D, HC.labels_)))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(D)
    print('kmeans Silhouette Score  {} '.format(metrics.silhouette_score(D, kmeans.labels_)))

    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full').fit(D)
    gmmlabels_ = gmm.predict(D)
    print('gmm Silhouette Score  {} '.format(metrics.silhouette_score(D, gmmlabels_)))
    
    fig, axs = plt.subplots(2, 2, figsize=(13, 7))
    axs[0, 0].scatter(D[:, 0], D[:, 1], cmap='viridis')
    axs[0, 0].set_title('Normal')

    axs[0, 1].scatter(D[:, 0], D[:, 1], c=gmmlabels_, cmap='viridis')
    axs[0, 1].set_title('GMM')

    axs[1, 0].scatter(D[:, 0], D[:, 1], c=kmeans.labels_, cmap='viridis')
    axs[1, 0].set_title('K-Means')

    axs[1, 1].scatter(D[:, 0], D[:, 1], c=HC.labels_, cmap='viridis')
    axs[1, 1].set_title('HC')
    plt.show()
    
    Labels.append(HC.labels_)
    Labels.append(kmeans.labels_)
    Labels.append(gmmlabels_)
    return Labels


def pltPathologyClusters(labels, path):
    # clusterimgDir = "../../Data/clusters_journal.PNG"
    # image = Image.open(clusterimgDir) 
    # plt.figure(figsize = (85,12))
    # plt.imshow(image)
    # plt.axis('off')
    
    sub_directories = [str(cluster) for cluster in set(labels)]
    displayImages = []
    
    for cluster in sub_directories:
        direct = path + '/{}'.format(cluster)
        if len(os.listdir(direct))-9 > 9: # if directory has less than 9 images set index to 0 else random index
            index = np.random.randint(9,len(os.listdir(direct))-9)
        else:
            index = 0 # pick the first 10 images
        clusterList = [] # reset the row
        for file in os.listdir(direct)[index:index+9]: # random sample of 9 images
            if file.endswith('.tif'):
                #image = Image.open(os.path.join(trainData, file)) 
                image = tifffile.imread(os.path.join(path, file))
                clusterList.append(image)
                displayImages.append(image) # list of ALL Images
        
    fig = plt.figure(figsize=(14, 14))
    
    columns = 9
    rows = len(sub_directories)
    for i in range(1, columns*rows+1):
        img = displayImages[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.show()
    

def symlink_rel(src, dst):
    rel_path_src = os.path.relpath(src, os.path.dirname(dst))
    os.symlink(rel_path_src, dst)

def clusterintoDirectories(labels, path, imagenamesList):
    sub_directories = [str(cluster) for cluster in set(labels)]

    for filename in os.listdir(path):
        if filename.endswith('.tif'):
            for cluster in sub_directories: # count of distinct elements = no. of clusters
                os.makedirs(path + '/{}'.format(cluster) , exist_ok=True)

    for i in range(len(imagenamesList)):
        # if there isnt already a symlink of this image in the coressponding subdirectory
        if imagenamesList[i] not in os.listdir(path + '/' + sub_directories[labels[i]]): 
            symlink_rel(path + '/{}'.format(imagenamesList[i]) , 
                       path + '/{}'.format(labels[i]) + '/' + imagenamesList[i])


def evaluatewithLabels(labels_pred, labels_true):
    print("Adjusted Rand index {}".format(metrics.adjusted_mutual_info_score(labels_pred, labels_true)))
    print("homogeneity_score {}".format(metrics.homogeneity_score(labels_true, labels_pred)))
    print("adjusted_rand_score {}".format(metrics.adjusted_rand_score(labels_true,labels_pred)))
    print("completeness_score {}".format(metrics.completeness_score(labels_true, labels_pred)))
    print("v_measure_score {}".format(metrics.v_measure_score(labels_true, labels_pred, beta=0.6)))

def evaluateAll3withLabels(Labels, labels_true):

    summ = 0
    algo_list = []

    for algo in Labels: # Labels has a length 3, HC is 0, KMeans is 1, GMM is 2
        summ += metrics.adjusted_mutual_info_score(algo, labels_true)
        summ += metrics.homogeneity_score(algo, labels_true)
        summ += metrics.adjusted_rand_score(algo, labels_true)
        summ += metrics.completeness_score(algo, labels_true)
        summ += metrics.v_measure_score(algo, labels_true)
        algo_list.append(summ / 5) # append the average
        summ = 0

    print(algo_list)

    idx = algo_list.index(max(algo_list))
    if idx == 0:
        algo_name = "HC"
    elif idx == 1:
        algo_name = "KMeans"
    elif idx == 2:
        algo_name = "GMM"

    print("{} Adjusted Rand index {}".format(algo_name,metrics.adjusted_mutual_info_score(labels_true, Labels[idx])))
    print("{} homogeneity_score {}".format(algo_name,metrics.homogeneity_score(labels_true, Labels[idx])))
    print("{} adjusted_rand_score {}".format(algo_name,metrics.adjusted_rand_score(Labels[idx],labels_true)))
    print("{} completeness_score {}".format(algo_name,metrics.completeness_score(labels_true, Labels[idx])))
    print("{} v_measure_score {}".format(algo_name,metrics.v_measure_score(labels_true, Labels[idx], beta=0.6)))



