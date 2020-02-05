import os

directory = '../K-MeansData'
c = 1

for filename in os.listdir(directory):
	print(filename)
	if not filename.startswith('.') and '.tif' in filename:
		os.rename(os.path.join(directory, filename), os.path.join(directory, 'Image {}'.format(c) + '.tif') )
		c += 1