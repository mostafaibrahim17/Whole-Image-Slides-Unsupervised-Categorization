from __future__ import print_function
import json
import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
from optparse import OptionParser
import re
import shutil
from unicodedata import normalize
import numpy as np
import scipy.misc
import subprocess
from glob import glob
from multiprocessing import Process, JoinableQueue
import time
import os
import sys
import dicom
from scipy.misc import imsave
from scipy.misc import imread
from scipy.misc import imresize

from xml.dom import minidom
from PIL import Image, ImageDraw




## CLI arguments note

python3 tile_slide.py <WSI path>

# the tile_slide script turns the WSI (.svs / jpg / dcm) into a directory with the magnifications (ranging from 20 to 1? )

# TO-DO: just got to the 20.0 folder and cluster on those tiles

# we can either default the clustering on 20X or the give the user the option (probably just default to 20)

# Clustering can be done on both a WSI (which the software package will then turn into tiles) or just a directory of tiles

# Supported extension for tiles: .tiff, .jpg .png .jpeg (.dcm .svs)?


# CLI arguments for software package

python3 tile_and_cluster.py <data path> <n_clusters> <svs / tiles> # if svs run tile_slide.py  <data path>, if tiles just cluster on the tiles at the data path

# TO-DO: calls tile_slide.py with the WSI path then

# TO-DO: Note for the autoencoders, need to resize the Images from 256 to 96
resized_image = cv2.resize(image, (96, 96)) 


# Q: Which approach to use by default for the software package...