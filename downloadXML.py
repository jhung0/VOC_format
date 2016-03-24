###
#For specified files, replace the files in the current folder with ones downloaded from the internet 
# python downloadXML.py /home/jyhung/Downloads/collection/Annotations/users/jane24/54gamplusring_20151223_outlines/
###

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


input_dir = argv[1]
extension = '.xml'
#download from this file forward
first_file_num = 241
for filename in os.listdir(input_dir):
    file_, file_extension = os.path.splitext(filename)
    print file_, file_extension
    if int(file_.split('-')[-1]) >= first_file_num:
      url = 'http://labelme2.csail.mit.edu/Release3.0/Annotations/users/jane24///54gamplusring_20151223_outlines/'+file_+extension
      try:
        os.remove(os.path.join(input_dir,file_+extension))
      except:
        print 'file not there'
      os.system('wget ' + url + ' -P ' + input_dir)
    '''
    #copy
    for fname in [file_ + extension, file_.upper() + extension, file_.lower() + extension, file_ + backup_extension]: 
        full_fname = os.path.join(replacement_dir, fname)
        if os.path.isfile(full_fname):
            shutil.copy(full_fname, input_dir)
            break
    else:
        #raise Exception('%s not found in directory'%(file_+extension))
        print '%s not found in directory'%(file_+extension)
        continue


    #delete
    try:
        os.remove(os.path.join(input_dir,filename))
    except:
        raise Exception('cannot remove %s'%filename)
    '''
