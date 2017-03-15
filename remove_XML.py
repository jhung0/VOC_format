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
remove certain XML files in LabelMe Annotation Tool
python remove_XML.py 301352 range(71,75)
'''
if len(argv)%2 == 0:
	raise Exception('needs even number of arguments')
for i in range(1, len(argv), 2):
	input_dir = argv[i]
	
for filename in os.listdir(input_dir):
    file_, file_extension = os.path.splitext(filename)
    print file_, file_extension

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

