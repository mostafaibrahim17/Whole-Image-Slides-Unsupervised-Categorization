# Generalized Categorisation of Digital Pathology Whole Image Slides using Unsupervised Learning

Paper: https://arxiv.org/pdf/2012.13955.pdf

Abstract:

This project aims to break down large pathology images into small tiles and then cluster those tiles into distinct groups without the knowledge of true labels, our analysis shows how difficult certain aspects of clustering tumorous and non-tumorous cells can be and also shows that comparing the results of different unsupervised approaches is not a trivial task. The project also provides a software package to be used by the digital pathology community, that uses some of the approaches developed to perform unsupervised unsupervised tile classification, which could then be easily manually labelled.

The project uses a mixture of techniques ranging from classical clustering algorithms such as K-Means and Gaussian Mixture Models to more complicated feature extraction techniques such as deep Autoencoders and Multi-loss learning. Throughout the project, we attempt to set a benchmark for evaluation using a few measures such as completeness scores and cluster plots.

Throughout our results we show that Convolutional Autoencoders manages to slightly outperform the rest of the approaches due to its powerful internal representation learning abilities. Moreover, we show that Gaussian Mixture models produce better results than K-Means on average due to its flexibility in capturing different clusters. We also show the huge difference in the difficulties of classifying different types of pathology textures.

# Overview
This is an overview of the software package:
1. It first uses Openslide to break down the WSI into non-overlapping tiles, then it navigates to the magnification level specified by the used, then if the user selects:

(a) Option 1: It extracts the mean RGB and H&E from each tile and uses a Gaussian Mixture Model to fit the data with n clusters specified (estimated by the user).

(b) Option 2: It applies 2 sets of dimensionality reduction algorithms before clustering, the first set is the encoding process of an Autoencoder network, the second set further reduces the dimensions by applying the PCA algorithm, after this feature extraction process has finished, it uses a Gaussian Mixture Model to fit the data.

Then it creates n cluster sub-directories in the specified magnification level directory where each sub-directory represents a distinct cluster.It then saves a symbolic link of each of the images corresponding to that cluster in that sub-directory. The symbolic links where a clever design choice, we did not want to copy the images because that would take more time and space. Also, the symbolic links are easily previewable.

The software package is essentially a Command Line Interface (CLI) argument parser, it gives users the flexibility to:
• Estimate the number of groups that might be potentially found in their dataset • Choose the magnification level that they want to cluster on
• Choose which of the 2 approaches they think might be best

# Software Package
First run
```
pip  install -r requirements.txt
```

then run
```
python main.py <datapath> <option> <n_datatypes> <magnification_level>
```
the main script will first call tile_slide.py which uses openslide to create a directory with various different magnifications of the tiled WSI. Then it will either call manual_features.py or AE.py depending on the option specified which will create a sub-directory IN the magnification level directory with the clusters sub-directories.

The package was tested with Python 3.7 on WSIs (.svs). 

If the algorithm can only find one class, it will print out a message say so and in that case please use the other option.

If you want to use the autoencoder option, for optimal performance you might need to retrain the autoencoders on your dataset. However, I have provided the weights for the trained autoencoders on histopathology images.
