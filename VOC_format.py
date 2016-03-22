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
Randomly takes X subsamples of full image and outputs in other folder
Usage: python VOC_format.py [Output directory] [Image files]
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
    if deleted or not label or label[0] == 'e':
        continue
        
    if not DIFFICULT and label[0] == 'd':
        label = 'uncertain'
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
	return xmin, ymin, xmax, ymax

#if file exists, remove
def removeIfExists(output_dir, subdir, name): 
    filename = os.path.join(output_dir, subdir, name)
    try:
        os.remove(filename)
    except OSError:
        pass
    return filename


DIFFICULT = True #whether there's a difficult tag
ROTATE = True #whether to double subimages with half subimages rotated by 90, 180, 270 counterclockwise
UNCERTAIN_CLASS = False #don't have uncertain class, either ignore or tag as difficult

output_dir = argv[1]#os.path.join('/Users', 'jyhung', 'Documents', 'VOC_format', 'data')
print 'output director', output_dir
num_subimages = 50
print 'number of subimages (not including rotations)', num_subimages
small_size = 224
print 'size of subimages (px)', small_size

#clear existing files
for name in ['Annotations', 'ImageSets', 'Images']:
    clear_dir = os.path.join(output_dir, name)
    for f in os.listdir(clear_dir):
        os.remove(os.path.join(clear_dir, f))

#if train.txt or test.txt exists, remove
for train_or_test in ['train', 'test']:
    filename_train = removeIfExists(output_dir, 'ImageSets', train_or_test+'.txt')

#for each image, subsample image and for each subimage, create associated file with bounding box and class information
filenum = 1
for filename in argv[2:]:
    #choose whether file is part of training or testing set
    if filenum < 200:
        train_or_test = 'train'
    else:
        train_or_test = 'test'
    filenum += 1
    filename_train = os.path.join(output_dir, 'ImageSets',train_or_test+'.txt')

    file_, file_extension = os.path.splitext(filename)
    print file_, file_extension
    img = Image.open(filename)

    width = img.size[0]
    height = img.size[1]

    #get associated xml file and parse for all objects
    path_collection = os.path.abspath(filename).split('/Images/')
    if os.path.isfile(os.path.join(path_collection[0], 'Annotations', path_collection[1].split(file_extension)[0]+'.xml')):
        filename_xml = os.path.join(path_collection[0], 'Annotations', path_collection[1].split(file_extension)[0]+'.xml')
    elif os.path.isfile(os.path.join(path_collection[0], 'Annotations', path_collection[1].split(file_extension)[0].upper()+'.xml')):
        filename_xml = os.path.join(path_collection[0], 'Annotations', path_collection[1].split(file_extension)[0].upper()+'.xml')
    elif os.path.isfile(os.path.join(path_collection[0], 'Annotations', path_collection[1].split(file_extension)[0].lower()+'.xml')):
        filename_xml = os.path.join(path_collection[0], 'Annotations', path_collection[1].split(file_extension)[0].lower()+'.xml')
    else:
        raise Exception('%s xml file not found'%(path_collection[1].split(file_extension)[0]))
        
    data = []
    
    tree = ET.parse(filename_xml)
    root = tree.getroot()
    
    
    for obj in root.findall('object'):
    	
        xmin, ymin, xmax, ymax, label, difficult = extractObjectData(obj)
        object_data = [xmin, ymin, xmax, ymax, label, difficult]
        data.append(object_data)

    
    if train_or_test  == 'train':
    	#if ROTATE, double the number of subimages
    	if ROTATE:
    		num_subimages = 2*num_subimages
    		
        for sub in range(num_subimages):
            print sub
            empty = True
            subname = os.path.basename(file_)+'_'+str(sub)
            
            #randomly choose top left corner of subimage
            randx = random.randint(0, width-small_size)
            randy = random.randint(0, height-small_size)
            print('top left corner coordinates:%s,%s'%(randx, randy))
            #save cropped image
            cropped = img.crop((randx, randy, randx+small_size, randy+small_size))
            
            #if ROTATE, after processing the original subimage, rotate by 90 and process
            if ROTATE and sub%2 == 1:
            	cropped = cropped.rotate(90)
            	
            #if Annotation file exists, remove
            filename_annotation = removeIfExists(output_dir, 'Annotations', subname+'.txt')
            
            	
            #write and save annotation file, only including data that are within the bounds of the subimage
            edge_data = []
            inside_data = []
            for object_data in data:
            	#print object_data
                adjusted_data = np.array(object_data[0:4]).copy()
                #adjust according to top left corner
                adjusted_data = adjusted_data - np.array([randx, randy, randx, randy])#map(operator.sub, map(int, adjusted_data), [randx, randy, randx, randy])
    
                #write if all coordinates are inside the subimage
                #adjust if 1 or 2 coordinates are inside
                adjusted_data_x = np.append(adjusted_data[0], adjusted_data[2])
                adjusted_data_y = np.append(adjusted_data[1], adjusted_data[3])
		if np.all(adjusted_data >= 0) and np.all(adjusted_data < small_size): #inside image
		#if object is uncertain, and there is no uncertain class, then don't consider the subimage
		if not UNCERTAIN_CLASS and object_data[4].lower() == 'uncertain':
		    break
		empty = False
		inside_data.append(adjusted_data)
		#if ROTATE, rotate adjusted_data
		if ROTATE:
			adjusted_data = np.array([adjusted_data[1], adjusted_data[0], adjusted_data[3], adjusted_data[2] ])
		with open(filename_annotation, 'a') as fp:
		    for datum in adjusted_data:
		        fp.write(str(datum)+' ')
		    print adjusted_data, object_data[4]
		    fp.write(str(object_data[-2])+' '+str(object_data[-1])+'\n')
            #if annotation file not empty
            #save cropped image name in train.txt file and cropped image
            else:
	        if not empty:
	            with open(filename_train, 'a') as fp:
                	fp.write(subname+'\n')
                    cropped.save(os.path.join(output_dir, 'Images', subname+file_extension))
    elif train_or_test == 'test': #full image with all annotations
        empty = True
        #if Annotation file exists, remove
        name = os.path.basename(file_)
        filename_annotation = removeIfExists(output_dir, 'Annotations', name+'.txt')
        
        for object_data in data:
            empty = False
            with open(filename_annotation, 'a') as fp:
                for datum in object_data:
                    fp.write(str(datum)+' ')
                fp.write('\n')     
        if not empty:
            with open(filename_train, 'a') as fp:
                fp.write(name+'\n')
            img.save(os.path.join(output_dir, 'Images', name+file_extension))
