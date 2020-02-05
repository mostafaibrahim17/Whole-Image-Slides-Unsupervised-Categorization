from multiprocessing.pool import ThreadPool
from keras.utils import OrderedEnqueuer
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator

class ImageGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, features, targets, n_classes=99, batch_size=32, shuffle=True, repeats=1):
        self.n_classes = n_classes
        self.n_vals = len(targets)
        # since we are using data agumentation we repeat the number of times we show each image.
        # we show the same original image but it can be rotated or flipper each time, so it is not the "same" image
        self.list_IDs = np.repeat(np.arange(self.n_vals), repeats)  
        self.batch_size = batch_size
        self.features = features
        self.shuffle = shuffle
        self.targets = targets
        self.targets_mc = keras.utils.to_categorical(targets, num_classes=self.n_classes)
        self.indexes = np.arange(len(self.list_IDs))
        self.pool = None

        self.agumentator = ImageDataGenerator(rescale=1./255)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if self.pool is None:
            self.pool = ThreadPool(6)
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation_threads(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = self.features[list_IDs_temp]
        y = self.targets_mc[list_IDs_temp]
        X = np.array(self.agumentator.flow(X, batch_size=self.batch_size, shuffle=self.shuffle).next())

        return X, y

    def __data_generation_threads(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = self.features[list_IDs_temp]
        y = self.targets_mc[list_IDs_temp]
        X = np.array(self.pool.map(lambda xi: self.agumentator.random_transform(self.agumentator.standardize(xi)), X))

        return X, y