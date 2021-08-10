'''
    File name: 0b_tileLoop_deepzoom.py
    Date created: March/2017

	Source:
	Tiling code inspired from
	https://github.com/openslide/openslide-python/blob/master/examples/deepzoom/deepzoom_tile.py
	which is Copyright (c) 2010-2015 Carnegie Mellon University
	The code has been extensively modified 

	Objective:
	Tile svs, jpg or dcm images with the possibility of rejecting some tiles based based on xml or jpg masks

	Be careful:
	Overload of the node - may have memory issue if node is shared with other jobs.
'''
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
import os
import sys
import dicom
from scipy.misc import imsave,imread, imresize
from tqdm import tqdm
from xml.dom import minidom
from PIL import Image, ImageDraw

VIEWER_SLIDE_NAME = 'slide'
Image.MAX_IMAGE_PIXELS = 933120000

class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds,quality, _Bkg, _ROIpc):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._slide = None
        self._Bkg = _Bkg
        self._ROIpc = _ROIpc

    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            #associated, level, address, outfile = data
            associated, level, address, outfile, format, outfile_bw, PercentMasked, SaveMasks, TileMask = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            #try:
            if True:
                try:
                    tile = dz.get_tile(level, address)
                    # A single tile is being read
                    #check the percentage of the image with "information". Should be above 50%
                    gray = tile.convert('L')
                    bw = gray.point(lambda x: 0 if x<220 else 1, 'F')
                    arr = np.array(np.asarray(bw))
                    avgBkg = np.average(bw)
                    bw = gray.point(lambda x: 0 if x<220 else 1, '1')
                    # check if the image is mostly background
                    if avgBkg <= (self._Bkg / 100.0):
                        # print("PercentMasked: %.6f, %.6f" % (PercentMasked, self._ROIpc / 100.0) )
                        # if an Aperio selection was made, check if is within the selected region
                        if PercentMasked >= (self._ROIpc / 100.0):
                            tile.save(outfile, quality=self._quality)
                            if bool(SaveMasks)==True:
                                height = TileMask.shape[0]
                                width = TileMask.shape[1]
                                TileMaskO = np.zeros((height,width,3), 'uint8')
                                maxVal = float(TileMask.max())
                                TileMaskO[...,0] = (TileMask[:,:].astype(float)  / maxVal * 255.0).astype(int)
                                TileMaskO[...,1] = (TileMask[:,:].astype(float)  / maxVal * 255.0).astype(int)
                                TileMaskO[...,2] = (TileMask[:,:].astype(float)  / maxVal * 255.0).astype(int)
                                TileMaskO = imresize(TileMaskO, (arr.shape[0], arr.shape[1],3))
                                TileMaskO[TileMaskO<10] = 0
                                TileMaskO[TileMaskO>=10] = 255
                                imsave(outfile_bw,TileMaskO) #(outfile_bw, quality=self._quality)

                    self._queue.task_done()
                except:
                    print(level, address)
                    print("image %s failed at dz.get_tile for level %f" % (self._slidepath, level))
                    self._queue.task_done()

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap, limit_bounds=self._limit_bounds)

class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz, basename, format, associated, queue, slide, basenameJPG, xmlfile, mask_type, xmlLabel, ROIpc, ImgExtension, SaveMasks, Mag):
        self._dz = dz
        self._basename = basename
        self._basenameJPG = basenameJPG
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._slide = slide
        self._xmlfile = xmlfile
        self._mask_type = mask_type
        self._xmlLabel = xmlLabel
        self._ROIpc = ROIpc
        self._ImgExtension = ImgExtension
        self._SaveMasks = SaveMasks
        self._Mag = Mag

    def run(self):
        self._write_tiles()
        self._write_dzi()

    def _write_tiles(self):
            ########################################3
            # nc_added
        #level = self._dz.level_count-1
        Magnification = 20
        tol = 2
        #get slide dimensions, zoom levels, and objective information
        Factors = self._slide.level_downsamples
        try:
            Objective = float(self._slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
            print(self._basename + " - Obj information found")
        except:
            print(self._basename + " - No Obj information found")
            print(self._ImgExtension)
            if ("jpg" in self._ImgExtension) | ("dcm" in self._ImgExtension):
                #Objective = self._ROIpc
                Objective = 1.
                Magnification = Objective
                print("input is jpg - will be tiled as such with %f" % Objective)
            else:
                return
        #calculate magnifications
        Available = tuple(Objective / x for x in Factors)
        #find highest magnification greater than or equal to 'Desired'
        Mismatch = tuple(x-Magnification for x in Available)
        AbsMismatch = tuple(abs(x) for x in Mismatch)
        if len(AbsMismatch) < 1:
          print(self._basename + " - Objective field empty!")
          return
        xml_valid = False
        # a dir was provided for xml files

        if True:
            #if self._xmlfile != '' && :
            print(self._xmlfile, self._ImgExtension)
            ImgID = os.path.basename(self._basename)
            xmldir = os.path.join(self._xmlfile, ImgID + '.xml')
            print("xml:")
            print(xmldir)
            if (self._xmlfile != '') & (self._ImgExtension != 'jpg') & (self._ImgExtension != 'dcm'):
                print("read xml file...")
                mask, xml_valid, Img_Fact = self.xml_read(xmldir, self._xmlLabel)
                if xml_valid == False:
                    print("Error: xml %s file cannot be read properly - please check format" % xmldir)
                    return
            elif (self._xmlfile != '')  & (self._ImgExtension == 'dcm'):
                print("check mask for dcm")
                mask, xml_valid, Img_Fact = self.jpg_mask_read(xmldir)
                # mask <-- read mask 
                #  Img_Fact <-- 1
                # xml_valid <-- True if mask file exists.
                if xml_valid == False:
                    print("Error: xml %s file cannot be read properly - please check format" % xmldir)
                    return

            print("current directory: %s" % self._basename)

            #return

            for level in tqdm(range(self._dz.level_count-1,-1,-1)):
                ThisMag = Available[0]/pow(2,self._dz.level_count-(level+1))
                if self._Mag > 0:
                    if ThisMag != self._Mag:
                        continue
                ########################################
                #tiledir = os.path.join("%s_files" % self._basename, str(level))
                tiledir = os.path.join("%s_files" % self._basename, str(ThisMag))
                if not os.path.exists(tiledir):
                    os.makedirs(tiledir)
                cols, rows = self._dz.level_tiles[level]
                if xml_valid:
                    print("xml valid")
              
                for row in range(rows):
                    for col in range(cols):
                        InsertBaseName = False
                        if InsertBaseName:
                          tilename = os.path.join(tiledir, '%s_%d_%d.%s' % (
                                          self._basenameJPG, col, row, self._format))
                          tilename_bw = os.path.join(tiledir, '%s_%d_%d_mask.%s' % (
                                          self._basenameJPG, col, row, self._format))
                        else:
                          tilename = os.path.join(tiledir, '%d_%d.%s' % (
                                          col, row, self._format))
                          tilename_bw = os.path.join(tiledir, '%d_%d_mask.%s' % (
                                          col, row, self._format))
                        if xml_valid:
                            Dlocation, Dlevel, Dsize = self._dz.get_tile_coordinates(level,(col, row))
                            Ddimension = tuple([pow(2,(self._dz.level_count - 1 - level)) * x for x in self._dz.get_tile_dimensions(level,(col, row))])
                            startIndY_current_level_conv = (int((Dlocation[1]) / Img_Fact))
                            endIndY_current_level_conv = (int((Dlocation[1] + Ddimension[1]) / Img_Fact))
                            startIndX_current_level_conv = (int((Dlocation[0]) / Img_Fact))
                            endIndX_current_level_conv = (int((Dlocation[0] + Ddimension[0]) / Img_Fact))
                            TileMask = mask[startIndY_current_level_conv:endIndY_current_level_conv, startIndX_current_level_conv:endIndX_current_level_conv]
                            PercentMasked = mask[startIndY_current_level_conv:endIndY_current_level_conv, startIndX_current_level_conv:endIndX_current_level_conv].mean() 

                            print(Ddimension, startIndY_current_level_conv, endIndY_current_level_conv, startIndX_current_level_conv, endIndX_current_level_conv)


                            if self._mask_type == 0:
                                # keep ROI outside of the mask
                                PercentMasked = 1.0 - PercentMasked

                            if PercentMasked > 0:
                                print("PercentMasked_p %.3f" % (PercentMasked))
                            else:
                                print("PercentMasked_0 %.3f" % (PercentMasked))

 
                        else:
                            PercentMasked = 1.0
                            TileMask = []

                        if not os.path.exists(tilename):
                            self._queue.put((self._associated, level, (col, row),
                                        tilename, self._format, tilename_bw, PercentMasked, self._SaveMasks, TileMask))
                        self._tile_done()

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count
        if count % 100 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (
                    self._associated or 'slide', count, total),
                    end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)

    def _write_dzi(self):
        with open('%s.dzi' % self._basename, 'w') as fh:
            fh.write(self.get_dzi())

    def get_dzi(self):
        return self._dz.get_dzi(self._format)


    def jpg_mask_read(self, xmldir):
        # Original size of the image
        ImgMaxSizeX_orig = float(self._dz.level_dimensions[-1][0])
        ImgMaxSizeY_orig = float(self._dz.level_dimensions[-1][1])
        # Number of centers at the highest resolution
        cols, rows = self._dz.level_tiles[-1]
        # Img_Fact = int(ImgMaxSizeX_orig / 1.0 / cols)
        Img_Fact = 1
        try:
            # xmldir: change extension from xml to *jpg   
            xmldir = xmldir[:-4] + "mask.jpg"
            # xmlcontent = read xmldir image
            xmlcontent = imread(xmldir)
            xmlcontent = xmlcontent - np.min(xmlcontent)
            mask = xmlcontent / np.max(xmlcontent)
            # we want image between 0 and 1
            xml_valid = True
        except:
            xml_valid = False
            print("error with minidom.parse(xmldir)")
            return [], xml_valid, 1.0

        return mask, xml_valid, Img_Fact


    def xml_read(self, xmldir, Attribute_Name):
        # Original size of the image
        ImgMaxSizeX_orig = float(self._dz.level_dimensions[-1][0])
        ImgMaxSizeY_orig = float(self._dz.level_dimensions[-1][1])
        # Number of centers at the highest resolution
        cols, rows = self._dz.level_tiles[-1]

        NewFact = max(ImgMaxSizeX_orig, ImgMaxSizeY_orig) / min(max(ImgMaxSizeX_orig, ImgMaxSizeY_orig),15000.0)
        Img_Fact = float(ImgMaxSizeX_orig) / 5.0 / float(cols)
       
        print("image info:")
        print(ImgMaxSizeX_orig, ImgMaxSizeY_orig, cols, rows) 
        try:
            xmlcontent = minidom.parse(xmldir)
            xml_valid = True
        except:
            xml_valid = False
            print("error with minidom.parse(xmldir)")
            return [], xml_valid, 1.0

        xy = {}
        xy_neg = {}
        labelIDs = xmlcontent.getElementsByTagName('Annotation')
        for labelID in labelIDs:
            if (Attribute_Name==[]) | (Attribute_Name==''):
                    isLabelOK = True
            else:
                try:
                    labeltag = labelID.getElementsByTagName('Attribute')[0]
                    if (Attribute_Name==labeltag.attributes['Value'].value):
                    #if (Attribute_Name==labeltag.attributes['Name'].value):
                        isLabelOK = True
                    else:
                        isLabelOK = False
                except:
                	isLabelOK = False
            if Attribute_Name == "non_selected_regions":
                isLabelOK = True

            #print("label ID, tag:")
            #print(labelID, Attribute_Name, labeltag.attributes['Name'].value)
            #if Attribute_Name==labeltag.attributes['Name'].value:
            if isLabelOK:
                regionlist = labelID.getElementsByTagName('Region')
                for region in regionlist:
                    vertices = region.getElementsByTagName('Vertex')
                    regionID = region.attributes['Id'].value
                    NegativeROA = region.attributes['NegativeROA'].value
                    if len(vertices) > 0:
                        #print( len(vertices) )
                        if NegativeROA=="0":
                            xy[regionID] = []
                            for vertex in vertices:
                                # get the x value of the vertex / convert them into index in the tiled matrix of the base image
                                x = int(round(float(vertex.attributes['X'].value) / NewFact))
                                y = int(round(float(vertex.attributes['Y'].value) / NewFact))
                                xy[regionID].append((x,y))
                                #print(vertex.attributes['X'].value, vertex.attributes['Y'].value, x, y )
    
                        elif NegativeROA=="1":
                            xy_neg[regionID] = []
                            for vertex in vertices:
                                # get the x value of the vertex / convert them into index in the tiled matrix of the base image
                                x = int(round(float(vertex.attributes['X'].value) / NewFact))
                                y = int(round(float(vertex.attributes['Y'].value) / NewFact))
                                xy_neg[regionID].append((x,y))
    

        img = Image.new('L', (int(ImgMaxSizeX_orig/NewFact), int(ImgMaxSizeY_orig/NewFact)), 0)
        for regionID in xy.keys():
            xy_a = xy[regionID]
            ImageDraw.Draw(img,'L').polygon(xy_a, outline=255, fill=255)
        for regionID in xy_neg.keys():
            xy_a = xy_neg[regionID]
            ImageDraw.Draw(img,'L').polygon(xy_a, outline=255, fill=0)
        mask = np.array(img)
        scipy.misc.toimage(mask).save(os.path.join(os.path.split(self._basename[:-1])[0], "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + ".jpeg")) 
        return mask / 255.0, xml_valid, NewFact

class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, slidepath, basename, format, tile_size, overlap,
                limit_bounds, quality, workers, with_viewer, Bkg, basenameJPG, xmlfile, mask_type, ROIpc, oLabel, ImgExtension, SaveMasks, Mag):
        if with_viewer:
            # Check extra dependency before doing a bunch of work
            import jinja2
        self._slide = open_slide(slidepath)
        self._basename = basename
        self._basenameJPG = basenameJPG
        self._xmlfile = xmlfile
        self._mask_type = mask_type
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._with_viewer = with_viewer
        self._Bkg = Bkg
        self._ROIpc = ROIpc
        self._dzi_data = {}
        self._xmlLabel = oLabel
        self._ImgExtension = ImgExtension
        self._SaveMasks = SaveMasks
        self._Mag = Mag

        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                limit_bounds, quality, self._Bkg, self._ROIpc).start()

    def run(self):
        self._run_image()
        if self._with_viewer:
            for name in self._slide.associated_images:
                self._run_image(name)
            self._write_html()
            self._write_static()
        self._shutdown()

    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        if associated is None:
            image = self._slide
            if self._with_viewer:
                basename = os.path.join(self._basename, VIEWER_SLIDE_NAME)
            else:
                basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap,limit_bounds=self._limit_bounds)
        tiler = DeepZoomImageTiler(dz, basename, self._format, associated,self._queue, self._slide, self._basenameJPG, self._xmlfile, self._mask_type, self._xmlLabel, self._ROIpc, self._ImgExtension, self._SaveMasks, self._Mag)
        tiler.run()
        self._dzi_data[self._url_for(associated)] = tiler.get_dzi()

    def _url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self._slugify(associated)
        return '%s.dzi' % base

    def _write_html(self):
        import jinja2
        env = jinja2.Environment(loader=jinja2.PackageLoader(__name__),autoescape=True)
        template = env.get_template('slide-multipane.html')
        associated_urls = dict((n, self._url_for(n))
                for n in self._slide.associated_images)
        try:
            mpp_x = self._slide.properties[openslide.PROPERTY_NAME_MPP_X]
            mpp_y = self._slide.properties[openslide.PROPERTY_NAME_MPP_Y]
            mpp = (float(mpp_x) + float(mpp_y)) / 2
        except (KeyError, ValueError):
            mpp = 0
        data = template.render(slide_url=self._url_for(None),slide_mpp=mpp,associated=associated_urls, properties=self._slide.properties, dzi_data=json.dumps(self._dzi_data))
        with open(os.path.join(self._basename, 'index.html'), 'w') as fh:
            fh.write(data)

    def _write_static(self):
        basesrc = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                'static')
        basedst = os.path.join(self._basename, 'static')
        self._copydir(basesrc, basedst)
        self._copydir(os.path.join(basesrc, 'images'),
                os.path.join(basedst, 'images'))

    def _copydir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                shutil.copy(srcpath, os.path.join(dest, name))

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()

def ImgWorker(queue):
	print("ImgWorker started")
	while True:
		cmd = queue.get()			
		if cmd is None:
			queue.task_done()
			break
		print("Execute: %s" % (cmd))
		subprocess.Popen(cmd, shell=True).wait()
		queue.task_done()

def xml_read_labels(xmldir):
        try:
            xmlcontent = minidom.parse(xmldir)
            xml_valid = True
        except:
            xml_valid = False
            print("error with minidom.parse(xmldir)")
            return [], xml_valid
        labeltag = xmlcontent.getElementsByTagName('Attribute')
        xml_labels = []
        for xmllabel in labeltag:
            #xml_labels.append(xmllabel.attributes['Name'].value)
            xml_labels.append(xmllabel.attributes['Value'].value)
        if xml_labels==[]:
            xml_labels = ['']
        print(xml_labels)
        return xml_labels, xml_valid 

if __name__ == '__main__':
	parser = OptionParser(usage='Usage: %prog [options] <slide>')
	parser.add_option('-L', '--ignore-bounds', dest='limit_bounds',
		default=True, action='store_false',
		help='display entire scan area')
	parser.add_option('-e', '--overlap', metavar='PIXELS', dest='overlap',
		type='int', default=1,
		help='overlap of adjacent tiles [1]')
	parser.add_option('-f', '--format', metavar='{jpeg|png}', dest='format',
		default='jpeg',
		help='image format for tiles [jpeg]')
	parser.add_option('-j', '--jobs', metavar='COUNT', dest='workers',
		type='int', default=4,
		help='number of worker processes to start [4]')
	parser.add_option('-o', '--output', metavar='NAME', dest='basename',
		help='base name of output file')
	parser.add_option('-Q', '--quality', metavar='QUALITY', dest='quality',
		type='int', default=90,
		help='JPEG compression quality [90]')
	parser.add_option('-r', '--viewer', dest='with_viewer',
		action='store_true',
		help='generate directory tree with HTML viewer')
	parser.add_option('-s', '--size', metavar='PIXELS', dest='tile_size',
		type='int', default=254,
		help='tile size [254]')
	parser.add_option('-B', '--Background', metavar='PIXELS', dest='Bkg',
		type='float', default=50,
		help='Max background threshold [50]; percentager of background allowed')
	parser.add_option('-x', '--xmlfile', metavar='NAME', dest='xmlfile',
		help='xml file if needed')
	parser.add_option('-m', '--mask_type', metavar='COUNT', dest='mask_type',
		type='int', default=1,
		help='if xml file is used, keep tile within the ROI (1) or outside of it (0)')
	parser.add_option('-R', '--ROIpc', metavar='PIXELS', dest='ROIpc',
		type='float', default=50,
		help='To be used with xml file - minimum percentage of tile covered by ROI (white)')
	parser.add_option('-l', '--oLabelref', metavar='NAME', dest='oLabelref',
		help='To be used with xml file - Only tile for label which contains the characters in oLabel')
	parser.add_option('-S', '--SaveMasks', metavar='NAME', dest='SaveMasks',
		default=False,
		help='set to yes if you want to save ALL masks for ALL tiles (will be saved in same directory with <mask> suffix)')
	parser.add_option('-t', '--tmp_dcm', metavar='NAME', dest='tmp_dcm',
		help='base name of output folder to save intermediate dcm images converted to jpg (we assume the patient ID is the folder name in which the dcm images are originally saved)')
	parser.add_option('-M', '--Mag', metavar='PIXELS', dest='Mag',
		type='float', default=-1,
		help='Magnification at which tiling should be done (-1 of all)')

	(opts, args) = parser.parse_args()
	try:
		slidepath = args[0]
	except IndexError:
		parser.error('Missing slide argument')
	if opts.basename is None:
		opts.basename = os.path.splitext(os.path.basename(slidepath))[0]
	if opts.xmlfile is None:
		opts.xmlfile = ''

	files = glob(slidepath)  
	ImgExtension = slidepath.split('*')[-1]
	print(files)
	print("***********************")
	
	files = sorted(files)
	for imgNb in range(len(files)):
		filename = files[imgNb]
		opts.basenameJPG = os.path.splitext(os.path.basename(filename))[0]
		print("processing: " + opts.basenameJPG + " with extension: " + ImgExtension)

		if ("dcm" in ImgExtension) :
			print("convert %s dcm to jpg" % filename)
			if opts.tmp_dcm is None:
				parser.error('Missing output folder for dcm>jpg intermediate files')
			elif not os.path.isdir(opts.tmp_dcm):
				parser.error('Missing output folder for dcm>jpg intermediate files')

			if filename[-3:] == 'jpg':
                            continue
			ImageFile=dicom.read_file(filename)
			im1 = ImageFile.pixel_array
			maxVal = float(im1.max())
			minVal = float(im1.min())
			height = im1.shape[0]
			width = im1.shape[1]
			image = np.zeros((height,width,3), 'uint8')
			image[...,0] = ((im1[:,:].astype(float) - minVal)  / (maxVal - minVal) * 255.0).astype(int)
			image[...,1] = ((im1[:,:].astype(float) - minVal)  / (maxVal - minVal) * 255.0).astype(int)
			image[...,2] = ((im1[:,:].astype(float) - minVal)  / (maxVal - minVal) * 255.0).astype(int)
			dcm_ID = os.path.basename(os.path.dirname(filename))
			opts.basenameJPG = dcm_ID + "_" + opts.basenameJPG
			filename = os.path.join(opts.tmp_dcm, opts.basenameJPG + ".jpg")
			print(filename)
			imsave(filename,image)

			output = os.path.join(opts.basename, opts.basenameJPG)

			try:
				DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, '', ImgExtension, opts.SaveMasks, opts.Mag).run()
			except:
				print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))

		elif opts.xmlfile != '':
			xmldir = os.path.join(opts.xmlfile, opts.basenameJPG + '.xml')
			print("xml:")
			print(xmldir)
			if os.path.isfile(xmldir):
				if opts.mask_type==1:
					xml_labels, xml_valid = xml_read_labels(xmldir)
					for oLabel in xml_labels:
						print("label is %s and ref is %s" % (oLabel, opts.oLabelref))
						if (opts.oLabelref in oLabel) or (opts.oLabelref==''):
							output = os.path.join(opts.basename, oLabel, opts.basenameJPG)
							if not os.path.exists(os.path.join(opts.basename, oLabel)):
								os.makedirs(os.path.join(opts.basename, oLabel))
							try:
								DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, oLabel, ImgExtension, opts.SaveMasks, opts.Mag).run()
							except:
								print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
				else:
					# Background
					oLabel = "non_selected_regions"
					output = os.path.join(opts.basename, oLabel, opts.basenameJPG)
					if not os.path.exists(os.path.join(opts.basename, oLabel)):
						os.makedirs(os.path.join(opts.basename, oLabel))
					try:
						DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, oLabel, ImgExtension, opts.SaveMasks, opts.Mag).run()
					except:
						print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))

			else:
				if (ImgExtension == ".jpg") | (ImgExtension == ".dcm") :
					print("Input image to be tiled is jpg or dcm and not svs - will be treated as such")
					output = os.path.join(opts.basename, opts.basenameJPG)
					try:
						DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, '', ImgExtension, opts.SaveMasks, opts.Mag).run()
					except:
						print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))


				else:
					print("No xml file found for slide %s.svs (expected: %s). Directory or xml file does not exist" %  (opts.basenameJPG, xmldir) )
					continue
		else:
			output = os.path.join(opts.basename, opts.basenameJPG)
			if os.path.exists(output + "_files"):
				print("Image %s already tiled" % opts.basenameJPG)
				continue
			try:
			#if True:
				DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, '', ImgExtension, opts.SaveMasks, opts.Mag).run()
			except:
				print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))

	print("Finished Tiling the slide, you can now find a directory with the tiles and their magnifications")
