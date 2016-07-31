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
Takes full images (from training set) from LabelMe format and outputs them in VOC format in other folder
All labelled objects are labelled cell, else background
Takes max(X subimages, # subimages to get to 2*(# of objects in image))
Usage: python VOC_format3.py [Output directory] [Image files] [Other directories]
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

#decide whether the file should be in training or test set
def chooseTrainOrTest(filenum, filename):
    #if filenum > -1:
    if 'g16_t1_up' in filename:
        return 'train'
    return 'test'

#if file exists, remove
def removeIfExists(output_dir, subdir, name): 
    filename = os.path.join(output_dir, subdir, name)
    try:
        os.remove(filename)
    except OSError:
        pass
    return filename


DIFFICULT = True #whether there's a difficult tag
FROTATE = False #whether to (in addition to original subimages), flip and rotate by 90, 180, 270 

output_dir = argv[1]#os.path.join('/Users', 'jyhung', 'Documents', 'VOC_format', 'data')
image_dir = argv[2]
other_dir = argv[3:]
print 'output director', output_dir
num_subimages = 25
print 'number of subimages (not including rotations)', num_subimages
small_size = 448
print 'size of subimages (px)', small_size

#clear existing files
slide_name = os.path.split(image_dir)[1]
other_slide_names = [os.path.split(other_dir[i])[1] for i in range(len(other_dir))]
all_slide_names = other_slide_names.extend(slide_name)
print slide_name, other_slide_names, all_slide_names

for name in ['Annotations', 'Images', 'ImageSets']:
	for s_name in all_slide_names:
		if name == 'ImageSets':
			clear_dir = os.path.join(output_dir, name)
		else:
			clear_dir = os.path.join(output_dir, name, slide_name)
		for f in os.listdir(clear_dir):
			os.remove(os.path.join(clear_dir, f))

#if train.txt or test.txt exists, remove
for train_or_test in ['train', 'test']:
    filename_train = removeIfExists(output_dir, 'ImageSets', train_or_test+'.txt')

#if FROTATE, double the number of subimages
if FROTATE:
    num_subimages = 8*num_subimages
    		
#for each image, subsample image and for each subimage, create associated file with bounding box and class information
filenum = 1
for current_dir in all_slide_names:
	current_dir = os.path.join(os.path.split(image_dir)[0],current_dir)
	for filename in os.listdir(current_dir):
	    file_, file_extension = os.path.splitext(filename)
	    s_name = os.path.split(current_dir)[1]
	    print file_, file_extension, s_name
	    img = Image.open(filename)
	
	    #choose whether file is part of training or testing set
	    train_or_test = chooseTrainOrTest(filenum, file_)
	    
	    filenum += 1
	    filename_train = os.path.join(output_dir, 'ImageSets',train_or_test+'.txt')
	
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
	    data_infected = []
	    for obj in root.findall('object'):
	    	try:
	    		xmin, ymin, xmax, ymax, label, difficult = extractObjectData(obj)
	        except:
	        	continue
	        
	        object_data = [xmin, ymin, xmax, ymax, label, difficult]
	        data.append(object_data)
	        if label in ['tro', 'sch', 'gam', 'ring']:
	        	data_infected.extend([1])
	        else:
	        	data_infected.extend([0])
	    
	    if train_or_test  == 'train':
	    	#infected_found = [0 for i in range(len(data))]
	    	total_num_objects = len(data)
	    	num_objects = 0
	    	sub = 0
	        while sub < num_subimages or num_objects < 2*total_num_objects:
	        	print num_objects, total_num_objects
	        	print sub
	        	sub += 1
	        	empty = True
	        	subname = os.path.basename(file_)+'_'+str(sub)
	            
			#randomly choose top left corner of subimage
			randx = random.randint(0, width-small_size)
			randy = random.randint(0, height-small_size)
			print('top left corner coordinates:%s,%s'%(randx, randy))
			#save cropped image
			cropped = img.crop((randx, randy, randx+small_size, randy+small_size))
			
			#if Annotation file exists, remove
			filename_annotation = removeIfExists(output_dir, 'Annotations', s_name, subname+'.txt')
			
			#if FROTATE, flip/rotate according to subimage number 
			if FROTATE:
				for ii in range(8,0,-1):
					if sub%8 == ii-1:
						for _ in range(0, int((sub%8)/2)):
							#print 'rotate'
							cropped = cropped.rotate(90)
				if sub%2 == 1:
					#print 'flip'
					cropped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
	
			#write and save annotation file, only including data that are within the bounds of the subimage
			for object_data in data:
				#print object_data
				adjusted_data = np.array(object_data[0:4]).copy()
				#adjust according to top left corner
				adjusted_data = adjusted_data - np.array([randx, randy, randx, randy])#map(operator.sub, map(int, adjusted_data), [randx, randy, randx, randy])
			
				#inside image
				if np.all(adjusted_data >= 0) and np.all(adjusted_data < small_size): 
					empty = False
					#if FROTATE, change adjusted_data
					if FROTATE:
						for ii in range(8,0,-1):
							if sub%8 == ii-1:
								for _ in range(0, int((sub%8)/2)):
									adjusted_data = np.array([adjusted_data[1], small_size - adjusted_data[2], adjusted_data[3], small_size - adjusted_data[0]])
						if sub%2 == 1:
							adjusted_data = np.array([small_size - adjusted_data[2], adjusted_data[1], small_size - adjusted_data[0],  adjusted_data[3]])
					num_objects += 1
					with open(filename_annotation, 'a') as fp:
						for datum in adjusted_data:
							fp.write(str(datum)+' ')
						print adjusted_data, object_data
						fp.write('cell'+' '+str(object_data[-1])+'\n')
		        #if annotation file not empty
	    		#save cropped image name in train.txt file and cropped image
		    	else:
		    		if not empty:
		    			with open(filename_train, 'a') as fp:
		    				fp.write(subname+'\n')
		    			cropped.save(os.path.join(output_dir, 'Images', s_name, subname+file_extension))
		    			#cropped.save(os.path.join(output_dir, 'Images', subname+file_extension))
	
