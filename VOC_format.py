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
'''

#if file exists, remove
def removeIfExists(output_dir, subdir, name): 
    filename = os.path.join(output_dir, subdir, name)
    try:
        os.remove(filename)
    except OSError:
        pass
    return filename
    
IGNORE_EDGE_CELLS = True
UNCERTAIN_CLASS = True
PATCH_EDGE_CELLS = True

output_dir = argv[1]#os.path.join('/Users', 'jyhung', 'Documents', 'VOC_format', 'data')
print 'output director', output_dir
num_subimages = 96
print 'number of subimages', num_subimages
small_size = 224
print 'size of subimages (px)', small_size

#clear existing files
for name in ['Annotations', 'ImageSets', 'Images']:
    clear_dir = os.path.join(output_dir, name)
    for f in os.listdir(clear_dir):
        os.remove(os.path.join(clear_dir, f))

#if train.txt or test.txt exists, remove
for train_or_test in ['train', 'text']:
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
    tree = ET.parse(filename_xml)
    root = tree.getroot()
    data = []
    for object in root.findall('object'):
    	deleted = int(object.find('deleted').text)
        label = object.find('name').text
        #if label is empty, skip
        #if IGNORE_EDGE_CELLS is true and label begins with e, skip object (CONTINUE NOT BREAK)
        #if label starts with e (cell is on the edge), relabel without e
        if deleted or not label:
            continue
        elif IGNORE_EDGE_CELLS and label[0] == 'e':
            continue
        elif label[0] == 'e':
            label = label[1:]
            
        if UNCERTAIN_CLASS and label[0] == 'd':
            label = 'uncertain'
        elif label[0] == 'd':
            label = label[1:]
        try:
            box = object.find('segm').find('box')
            x = []
            y = []
            x.append(int(box.find('xmin').text))
            y.append(int(box.find('ymin').text))
            x.append(int(box.find('xmax').text))
            y.append(int(box.find('ymax').text))

        except:
            polygon = object.find('polygon')
            x = []
            y = []
            for pt in polygon.findall('pt'):
                x.append(int(pt.find('x').text))
                y.append(int(pt.find('y').text))
        xmin = int(min(x))
        ymin = int(min(y))
        xmax = int(max(x))
        ymax = int(max(y))

        object_data = [xmin, ymin, xmax, ymax, label]
        if object_data[0] >= object_data[2] or object_data[1] >= object_data[3]:
            raise Exception('object data ', object_data)
        data.append(object_data)

    
    if train_or_test  == 'train':
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
    
            #if Annotation file exists, remove
            filename_annotation = removeIfExists(output_dir, 'Annotations', subname+'.txt')
    
            #write and save annotation file, only including data that are within the bounds of the subimage
	    edge_data = []
	    inside_data = []
            for object_data in data:
            	print object_data
                adjusted_data = np.array(object_data[0:-1]).copy()
                #adjust according to top left corner
                adjusted_data = adjusted_data - np.array([randx, randy, randx, randy])#map(operator.sub, map(int, adjusted_data), [randx, randy, randx, randy])
    
                #write if all coordinates are inside the subimage
                #adjust if 1 or 2 coordinates are inside
                adjusted_data_x = np.append(adjusted_data[0], adjusted_data[2])
                adjusted_data_y = np.append(adjusted_data[1], adjusted_data[3])
                if IGNORE_EDGE_CELLS:
                    if np.all(adjusted_data >= 0) and np.all(adjusted_data < small_size): #inside image
                        empty = False
			inside_data.append(adjusted_data)
                        with open(filename_annotation, 'a') as fp:
                            for datum in adjusted_data:
                                fp.write(str(datum)+' ')
                            print object_data, adjusted_data, object_data[-1]
                            fp.write(str(object_data[-1])+'\n')
		    elif PATCH_EDGE_CELLS and not (np.all(adjusted_data_x < 0) or np.all(adjusted_data_y >= small_size) or
                                np.all(adjusted_data_x >= small_size) or np.all(adjusted_data_y < 0)): #edge of image
			if np.any(adjusted_data >= small_size):
                            adjusted_data = np.minimum(adjusted_data, small_size-1)
		        if np.any(adjusted_data < 0):
		            adjusted_data = np.maximum(adjusted_data, 0)
			edge_data.append(adjusted_data)
                else:
                    if not (np.all(adjusted_data_x < 0) or np.all(adjusted_data_y >= small_size) or
                                np.all(adjusted_data_x >= small_size) or np.all(adjusted_data_y < 0)):
                        with open(filename_annotation, 'a') as fp:
                            if np.any(adjusted_data >= small_size):
                                adjusted_data = np.minimum(adjusted_data, small_size-1)#map(min, adjusted_data, [size-1]*4)
                            if np.any(adjusted_data < 0):
                                adjusted_data = np.maximum(adjusted_data, 0)#map(max, adjusted_data, [0]*4)
                            for datum in adjusted_data:
                                fp.write(str(datum)+' ')
                            print object_data, adjusted_data, object_data[-1]
                            fp.write(str(object_data[-1])+'\n')
                            empty = False

	    #alter cropped image 
	    if PATCH_EDGE_CELLS:
	    	cropped = np.asarray(cropped).copy()
		cropped2 = cropped.copy()
		#mode of each channel
		mode_ = [stats.mode(cropped2[:,:,0], axis=None)[0], stats.mode(cropped2[:,:,1], axis=None)[0], stats.mode(cropped2[:,:,2])[0]]

		for edge_object in edge_data:
		    patch_shape = cropped2[edge_object[1]:edge_object[3], edge_object[0]:edge_object[2], :].shape
		    patch = np.ones(patch_shape)
		    for i in range(3):
		        #patch[:,:,i] = 255./2 + 255*(np.random.random_sample(patch[:,:,i].shape) - 0.5)
		        patch[:,:,i] = mode_[i] + min(mode_[i], 255-mode_[i])*.2*(np.random.random_sample(patch[:,:,i].shape) - 0.5)
		    cropped[edge_object[1]:edge_object[3],edge_object[0]:edge_object[2], :] = patch
		    for inside_object in inside_data:
			ixmin, iymin = max(edge_object[0], inside_object[0]), max(edge_object[1], inside_object[1])
			ixmax, iymax = min(edge_object[2], inside_object[2]), min(edge_object[3], inside_object[3])
			if ixmin >= ixmax or iymin >= iymax:
			    continue
			else: #replace overlapping region with original
			    overlap_box = np.array([ixmin, iymin, ixmax, iymax])
			    cropped[overlap_box[1]:overlap_box[3],overlap_box[0]:overlap_box[2],:] = cropped2[overlap_box[1]:overlap_box[3], overlap_box[0]:overlap_box[2],:]
		cropped = Image.fromarray(cropped, 'RGB')
		
            #if annotation file not empty
            #save cropped image name in train.txt file and cropped image
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
