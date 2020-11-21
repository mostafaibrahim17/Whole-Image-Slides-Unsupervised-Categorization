def evaluateRegression():
    labels_pred = autoencoder.predict(meanMatrix)[1]
    for i in range(len(labels_pred)):
        if labels_pred[i] < 0.5:
            labels_pred[i] = 0
        else:
            labels_pred[i] = 1

    evaluation = metrics.classification_report(testlabels, labels_pred)
    print(evaluation)


def produceTrueLabels():
	csvFilePath = "../Data/regrData/train_labels.csv"
	df = pd.read_csv(csvFilePath)
	testlabels = []

	for image in image_names:
	    label_index = df[df["id"] == image.split(".")[0]].index[0]
	    testlabels.append(df["label"][label_index])
    

def pltPathologyClusters(labels):
    clusterimgDir = "../Data/clusters_journal.PNG"
    image = Image.open(clusterimgDir) 
    plt.figure(figsize = (80,10))
    plt.imshow(image)
    plt.axis('off')
    
    sub_directories = [str(cluster) for cluster in set(labels)]
    displayImages = []
    
    for cluster in sub_directories:
        direct = trainData + '/{}'.format(cluster)
        index = np.random.randint(0,len(os.listdir(direct)))
        for file in os.listdir(direct)[index:index+9]: # random sample of 9 images
            if file.endswith('.tif'):
                image = Image.open(os.path.join(direct, file)) 
                displayImages.append(np.asarray( image, dtype="uint8" ))
    
    fig = plt.figure(figsize=(14, 14))
    
    columns = 9
    rows = len(sub_directories)
    print(len(displayImages))
    print(columns*rows+1)
    j = 0
    for i in range(1, columns*rows+1):
        img = displayImages[j]
        j+=1
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.text(rows, columns, "ss")
        plt.axis('off')
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.show()
    
def ClusterAndPlot(n_clusters):
    Labels = []
    
    print(X.shape)
    HC = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward').fit(X)
    print('HC Silhouette Score  {} '.format(metrics.silhouette_score(X, HC.labels_)))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    print('kmeans Silhouette Score  {} '.format(metrics.silhouette_score(X, kmeans.labels_)))

    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full').fit(X)
    gmmlabels_ = gmm.predict(X)
    print('gmm Silhouette Score  {} '.format(metrics.silhouette_score(X, gmmlabels_)))
    
    fig, axs = plt.subplots(2, 2, figsize=(13, 7))
    axs[0, 0].scatter(X[:, 0], X[:, 1], cmap='viridis')
    axs[0, 0].set_title('Normal')

    axs[0, 1].scatter(X[:, 0], X[:, 1], c=gmmlabels_, cmap='viridis')
    axs[0, 1].set_title('GMM')

    axs[1, 0].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
    axs[1, 0].set_title('K-Means')

    axs[1, 1].scatter(X[:, 0], X[:, 1], c=HC.labels_, cmap='viridis')
    axs[1, 1].set_title('HC')
    plt.show()
    
    Labels.append(HC.labels_)
    Labels.append(kmeans.labels_)
    Labels.append(gmmlabels_)
    return Labels


def symlink_rel(src, dst):
    rel_path_src = os.path.relpath(src, os.path.dirname(dst))
    os.symlink(rel_path_src, dst)

def clusterintoDirectories(labels):
    directory = trainData
    sub_directories = [str(cluster) for cluster in set(labels)]

    for filename in os.listdir(directory):
        if filename.endswith('.tif'):
            for cluster in sub_directories: # count of distinct elements = no. of clusters
                os.makedirs(directory + '/{}'.format(cluster) , exist_ok=True)

    for i in range(len(image_names)):
        # if there isnt already a symlink of this image in the coressponding subdirectory
        if image_names[i] not in os.listdir(directory + '/' + sub_directories[labels[i]]): 
            symlink_rel(directory + '/{}'.format(image_names[i]) , 
                       directory + '/{}'.format(labels[i]) + '/' + image_names[i])











