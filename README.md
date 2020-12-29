# Generalized Categorisation of Digital Pathology Whole Image Slides using Unsupervised Learning

Paper: https://arxiv.org/pdf/2012.13955.pdf

Abstract:

This project aims to break down large pathology images into small tiles and then cluster those tiles into distinct groups without the knowledge of true labels, our analysis shows how difficult certain aspects of clustering tumorous and non-tumorous cells can be and also shows that comparing the results of different unsupervised approaches is not a trivial task. The project also provides a software package to be used by the digital pathology community, that uses some of the approaches developed to perform unsupervised unsupervised tile classification, which could then be easily man- ually labelled.

The project uses a mixture of techniques ranging from classical clustering algorithms such as K-Means and Gaussian Mixture Models to more complicated feature extraction techniques such as deep Autoencoders and Multi-loss learning. Throughout the project, we attempt to set a benchmark for evaluation using a few measures such as completeness scores and cluster plots.

Throughout our results we show that Convolutional Autoencoders manages to slightly out- perform the rest of the approaches due to its powerful internal representation learning abilities. Moreover, we show that Gaussian Mixture models produce better results than K-Means on average due to its flexibility in capturing different clusters. We also show the huge difference in the difficulties of classifying different types of pathology textures.

# Instructions

The CLI takes several arguments:
1. Datapath
2. Options: Integer option: 1 or 2
(a) Option 1: Chooses the manually selected features approach for clustering (b) Option 2: Chooses the convolutional autoencoder approach, then clusters
3. Number of Clusters: Expected number of data types (Estimated by the user)
4. Magnification Level: The openslide package tiles the WSI into several magnification levels such as 1.25, 2.5, 5.0, 10.0 and 20.0, the user is given an option to choose any of those magnification levels.
