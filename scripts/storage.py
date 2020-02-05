## Data loading 

# directory = "new100"
trainData = "train"
testData = "test"
# x_train is a np array of 2D matrices where each matrix is the
# pixel values of 28 x 28 greyscale image
# (x_train, _), (x_test, _) = mnist.load_data()

new_train = []
new_test = []

for filename in os.listdir(trainData):
    if filename.endswith('.tif'):
        image = Image.open(os.path.join(trainData, filename)) 
        new_train.append(np.asarray( image, dtype="uint8" ))
        print("train appending {}".format(filename))

for filename in os.listdir(testData):
    if filename.endswith('.tif'):
        image = Image.open(os.path.join(testData, filename)) 
        new_test.append(np.asarray( image, dtype="uint8" ))
        print("test appending {}".format(filename))

## Data preprocessing

x_train = np.asarray(new_train)
x_test = np.asarray(new_test)
# normalize all values between 0 and 1 and flatten 28x28 images into vectors of size 784
# Basically switch from 2D matrices of 28 * 28 (and type int8) into 1D arrays of 784 (and type float32)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# npArray.shape returns dimensions (3), take the second 2 dimensions (28,28) and multliply them
# reshape Returns an array containing the same data with a new shape. (60000, 28, 28) -> (60000, 784)
x_train = np.reshape(x_train, (len(x_train), 96, 96, 3))
x_test = np.reshape(x_test, (len(x_test), 96, 96, 3))
np.prod(x_train.shape[1:])




































