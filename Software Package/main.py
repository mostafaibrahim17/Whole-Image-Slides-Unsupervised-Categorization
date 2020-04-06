import argparse
import subprocess
import os

parser = argparse.ArgumentParser(description='Categorize this Whole Image Slide Into Tiles')
parser.add_argument('datapath', help='datapath for 1 WSI')
parser.add_argument('option', type=int, help='1 is clustering using H&E values, 2 is clustering using Autoencoder compression')
parser.add_argument('n_datatypes', help='estimated number of distinct types of tissue in this WSI')
parser.add_argument('magnification_level', help='options are 1.25, 2.5, 5.0, 10.0 or 20.0')

args=parser.parse_args()


if __name__ == '__main__':
	print(args.datapath)

	if(args.datapath): # run tile_sile.py on it
		subprocess.call(["python3","tile_slide.py", args.datapath])

	print(args.option)
	if (args.option == 1): # manual features
		subprocess.call(["python3","manual.py", args.datapath, args.n_datatypes, args.magnification_level])
	elif (args.option == 2): # AEs
		subprocess.call(["python3","AE.py", args.datapath, args.n_datatypes, args.magnification_level])













