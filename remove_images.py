from glob import glob
from sys import argv
import fileinput
from PIL import Image
import random
import os
import xml.etree.ElementTree as ET
import operator
import numpy as np
import shutil
'''
Before any change in format, run this in image directory to replace images from images in the input directory
(like replacing .jpg images with .tif images with the same name)
Usage: python remove_images.py [directory where .tif images are] [directory where images to be replaced are]
'''
replacement_dir = argv[1]
extension = '.tif'
#for each image in the current directory replace it with an image of the same name (different extension) from the replacement directory
input_dir = argv[2]

for filename in os.listdir(input_dir):
    file_, file_extension = os.path.splitext(filename)
    print file_, file_extension

    #copy
    fname = os.path.join(replacement_dir, file_ + extension)
    if os.path.isfile(fname):
        shutil.copy(fname, input_dir)
    elif os.path.isfile(os.path.join(replacement_dir, file_.upper() + extension)):
        shutil.copy(os.path.join(replacement_dir, file_.upper() + extension), input_dir)
    else:
        #raise Exception('%s not found in directory'%(file_+extension))
        print '%s not found in directory'%(file_+extension)
        continue


    #delete
    try:
        os.remove(os.path.join(input_dir,filename))
    except:
        raise Exception('cannot remove %s'%filename)

