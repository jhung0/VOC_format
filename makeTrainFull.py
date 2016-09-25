from shutil import copyfile
from glob import glob
from sys import argv
import fileinput
from PIL import Image
import random
import os
import xml.etree.ElementTree as ET
import operator
import numpy as np
from scipy import stats
'''
makes trainfull.txt, which contains full images corresponding to train.txt
Usage: python makeTrainFull.py
Includes flags for different options. 
'''
#extract data from xml file
def extractObjectData(obj):
    deleted = int(obj.find('deleted').text)
    label = obj.find('name').text
    difficult = False
    #if label is empty, skip
    #if IGNORE_EDGE_CELLS is true and label begins with e, skip object (CONTINUE NOT BREAK)
    #if label starts with e (cell is on the edge), relabel without e
    if deleted or (not label) or (label[0] == 'e'):
    	raise Exception
        
    if not DIFFICULT and label[0] == 'd':
        label = 'uncertain'
    elif label == 'a':
	label = 'gam'
        difficult = True
    elif label[0] == 'd':
        difficult = True
        label = label[1:]
    try:
        box = obj.find('segm').find('box')
        x = [int(box.find('xmin').text), int(box.find('xmax').text)]
        y = [int(box.find('ymin').text), int(box.find('ymax').text)]
    except:
        polygon = obj.find('polygon')
        x = []
        y = []
        for pt in polygon.findall('pt'):
        	for px in pt.findall('x'):
        		x.extend([int(float(px.text))])
        	for py in pt.findall('y'):
        		y.extend([int(float(py.text))])
    xmin = int(min(x))
    ymin = int(min(y))
    xmax = int(max(x))
    ymax = int(max(y))
    if xmin >= xmax or ymin >= ymax:
        raise Exception('object data ', xmin, ymin, xmax, ymax)
    return xmin, ymin, xmax, ymax, label, difficult


#if file exists, remove
def removeIfExists(output_dir, subdir, name): 
    filename = os.path.join(output_dir, subdir, name)
    try:
        os.remove(filename)
    except OSError:
        pass
    return filename


DIFFICULT = True #whether there's a difficult tag
slide_name = 'g8_t1_up'
input_dir = os.path.join('/home/ubuntu/try1/data/')
labelme_dir = os.path.join('/var/www/html/LabelMeAnnotationTool/')
print 'input directory', input_dir
print 'slide name ', slide_name

#go into train.txt, create a set of each image name
with open(os.path.join(input_dir, 'ImageSets', 'train.txt'), 'r+') as fp:
	lines = fp.readlines()
	file_set = set()
	for line in lines:
		line = line.rsplit('_', 1)[0]
		#print line
		file_set.add(line)

#make trainfull.txt
#copy appropriate images to Images/ and annotations to Annotations/
with open(os.path.join(input_dir, 'ImageSets', 'trainfull.txt'), 'w+') as fp:
	for line in file_set:
		fp.write(line + '\n')
print 'file set', file_set		

#for each file, save image and annotation
file_extension = '.jpg'
for filename in file_set: 
    print filename   
    #get associated xml file and parse for all objects
    path_collection = os.path.join(labelme_dir, 'Images')
    path_annotations = os.path.join(labelme_dir, 'Annotations')
    #for name in [filename, filename.upper(), filename.lower()]:
    try_filename_xml = os.path.join(path_annotations, filename+'.xml')
    print try_filename_xml
    if os.path.isfile(try_filename_xml):
    		filename_xml = try_filename_xml
    else:
        raise Exception('%s xml file not found'%(filename))
    print filename_xml    
    data = []
    tree = ET.parse(filename_xml)
    root = tree.getroot()
    for obj in root.findall('object'):
    	try:
    		xmin, ymin, xmax, ymax, label, difficult = extractObjectData(obj)
        except:
        	continue
        object_data = [xmin, ymin, xmax, ymax, label, difficult]
        data.append(object_data)
    filename_annotation = os.path.join(input_dir, 'Annotations', filename+'.txt')
        
    for object_data in data:
            with open(filename_annotation, 'a') as fp:
                for datum in object_data:
                    fp.write(str(datum)+' ')
                fp.write('\n')     
    print os.path.join(labelme_dir, 'Images', filename+file_extension)
    print os.path.join(input_dir, 'Images', filename+file_extension) 
    copyfile(os.path.join(labelme_dir, 'Images', filename+file_extension), os.path.join(input_dir, 'Images', filename+file_extension))
    #img.save(os.path.join(output_dir, 'Images', slide_name, name+file_extension))
