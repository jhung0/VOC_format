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
Takes full images from LabelMe format and outputs them in VOC format in other folder
Usage: python VOC_format2.py [Output directory] [Image files]

'''

#extract data from xml file
def extractObjectData(obj):
    deleted = int(obj.find('deleted').text)
    label = obj.find('name').text
    difficult = False
    #if label is empty, skip
    if deleted or (not label):
    	raise Exception

    if not DIFFICULT and label[0] == 'd':
        label = 'uncertain'
    elif label == 'a':
        difficult = True
    elif label[0] == 'd':
        difficult = True
        label = label[1:]
    try:
        box = obj.find('segm').find('box')
        x = [int(box.find('xmin').text), int(box.find('xmax').text)]
        y = [int(box.find('ymin').text), int(box.find('ymax').text)]
    except:
        polygon = object.find('polygon')
        x = [int(pt.find('x').text) for pt in polygon.findall('pt')]
        y = [int(pt.find('y').text) for pt in polygon.findall('pt')]

    xmin = int(min(x))
    ymin = int(min(y))
    xmax = int(max(x))
    ymax = int(max(y))

    if xmin >= xmax or ymin >= ymax:
        raise Exception('object data ', xmin, ymin, xmax, ymax)
    return xmin, ymin, xmax, ymax, label, difficult

#decide whether the file should be in training or test set
def chooseTrainOrTest(filenum, filename):
    if filenum <= 31:
        return 'train'
    elif filenum > 31:
        return 'test'
    else:
        return 'none'

#if file exists, remove
def removeIfExists(output_dir, subdir, name):
    filename = os.path.join(output_dir, subdir, name)
    try:
        os.remove(filename)
    except OSError:
        pass
    return filename


DIFFICULT = True #whether there's a difficult tag

output_dir = argv[1]#os.path.join('/Users', 'jyhung', 'Documents', 'VOC_format', 'data')
print 'output director', output_dir

#clear existing files
for name in ['Annotations', 'ImageSets', 'Images']:
    clear_dir = os.path.join(output_dir, name)
    for f in os.listdir(clear_dir):
        os.remove(os.path.join(clear_dir, f))

#if train.txt or test.txt exists, remove
for train_or_test in ['train', 'test']:
    filename_imageset = removeIfExists(output_dir, 'ImageSets', train_or_test + '.txt')

#for each image, subsample image and for each subimage, create associated file with bounding box and class information
filenum = 1
for filename in argv[2:]:
    file_, file_extension = os.path.splitext(filename)
    print file_, file_extension
    img = Image.open(filename)

    #choose whether file is part of training or testing set
    train_or_test = chooseTrainOrTest(filenum, file_)

    filenum += 1
    filename_imageset = os.path.join(output_dir, 'ImageSets', train_or_test + '.txt')

    width = img.size[0]
    height = img.size[1]

    #get associated xml file and parse for all objects
    path_collection = os.path.abspath(filename).split('/Images/')
    path_annotations = os.path.join(path_collection[0], 'Annotations')
    xml_name = path_collection[1].split(file_extension)[0]
    for name in [xml_name, xml_name.upper(), xml_name.lower()]:
    	try_filename_xml = os.path.join(path_annotations, name+'.xml')
    	if os.path.isfile(try_filename_xml):
    		filename_xml = try_filename_xml
    		break
    else:
        raise Exception('%s xml file not found'%(xml_name))

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

    #full image with all annotations
    empty = True
    #if Annotation file exists, remove
    name = os.path.basename(file_)
    filename_annotation = removeIfExists(output_dir, 'Annotations', name+'.txt')

    for object_data in data:
        empty = False
        #if test (where test images are not annotated), don't include annotations
        if train_or_test == 'test':
            break
        with open(filename_annotation, 'a') as fp:
            for datum in object_data:
                fp.write(str(datum)+' ')
            if train_or_test == 'train':
                fp.write(str(object_data[-2])+' '+str(object_data[-1])+'\n')
            else:
                fp.write('\n')

    if not empty:
        with open(filename_imageset, 'a') as fp:
            fp.write(name+'\n')
        img.save(os.path.join(output_dir, 'Images', name+file_extension))
