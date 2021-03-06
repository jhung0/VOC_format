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

DIFFICULT = True #whether there's a difficult tag
FROTATE = True #whether to (in addition to original subimages), flip and rotate by 90, 180, 270 
UNCERTAIN_CLASS = False #don't have uncertain class, either ignore or tag as difficult

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

#decide whether the file should be in training or test set
def chooseTrainOrTest(rr, filename):
    if rr < 0.8:
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

output_dir = argv[1]#os.path.join('/Users', 'jyhung', 'Documents', 'VOC_format', 'data')
input_dir = argv[2]
print 'output directory', output_dir
print 'input directory', input_dir
num_subimages = 50
print 'number of subimages (not including rotations)', num_subimages
small_size = 448 
print 'size of subimages (px)', small_size
slide_name = os.path.basename(os.path.dirname(input_dir))
print 'slide name ', slide_name

#clear existing files
for name in ['Annotations', 'ImageSets', 'Images']:
    if name == 'ImageSets':
	clear_dir = os.path.join(output_dir, name)
    else:
	clear_dir = os.path.join(output_dir, name, slide_name)
    for f in os.listdir(clear_dir):
	os.remove(os.path.join(clear_dir, f))
    '''
    clear_dir = os.path.join(output_dir, name)
    for f in os.listdir(clear_dir):
        os.remove(os.path.join(clear_dir, f))
    '''

#if train.txt or test.txt exists, remove
for train_or_test in ['train', 'test']:
    filename_train = removeIfExists(output_dir, 'ImageSets', train_or_test+'.txt')

#if FROTATE, increase the number of subimages
if FROTATE:
    num_subimages = 8*num_subimages
    		
#for each image, subsample image and for each subimage, create associated file with bounding box and class information
filenum = 1
for filename in os.listdir(input_dir):
    filename = os.path.join(input_dir, filename)
    file_, file_extension = os.path.splitext(filename)
    print file_, file_extension
    img = Image.open(filename)

    #choose whether file is part of training or testing set
    train_or_test = chooseTrainOrTest(random.random(), file_)
    
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
    if train_or_test  == 'train':
	total_num_objects = len(data)
	if FROTATE:
		total_num_objects = 8*total_num_objects
        print 'total objects', total_num_objects
	num_objects = 0
	sub = 0
	while sub < num_subimages or num_objects < 2*total_num_objects:
        #for sub in range(num_subimages):
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
		filename_annotation = removeIfExists(output_dir, 'Annotations/'+slide_name, subname+'.txt')
		
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
		#only save annotation file if there's at least 1 object that is not difficult
		adjusted_diff = [] #list of difficult objects in image
		for object_data in data:
			adjusted_data = np.array(object_data[0:4]).copy()
			#adjust according to top left corner
			adjusted_data = adjusted_data - np.array([randx, randy, randx, randy])#map(operator.sub, map(int, adjusted_data), [randx, randy, randx, randy])
			#print adjusted_data	
			#inside image
			if np.all(adjusted_data >= 0) and np.all(adjusted_data < small_size): 
				#if FROTATE, change adjusted_data
				if FROTATE:
					for ii in range(8,0,-1):
						if sub%8 == ii-1:
							for _ in range(0, int((sub%8)/2)):
								adjusted_data = np.array([adjusted_data[1], small_size - adjusted_data[2], adjusted_data[3], small_size - adjusted_data[0]])
					if sub%2 == 1:
						adjusted_data = np.array([small_size - adjusted_data[2], adjusted_data[1], small_size - adjusted_data[0],  adjusted_data[3]])
				num_objects += 1
				if object_data[-1] == False:
					empty = False
					with open(filename_annotation, 'a') as fp:
						for datum in adjusted_data:
							fp.write(str(datum)+' ')
						print adjusted_data, object_data[4]
						fp.write(str(object_data[-2])+' '+str(object_data[-1])+'\n')
				else:
					#save in case difficult ones need to be added
					adjusted_diff.append([adjusted_data, object_data[-2], object_data[-1]])
					
	        #if annotation file not empty
		#add difficult objects to annotation file
    		#save cropped image name in train.txt file and cropped image
	    	else:
	    		if not empty:
				with open(filename_annotation, 'a') as fp:
                                        for diff_obj in adjusted_diff:
						for i in range(len(diff_obj[0])):
                                                	fp.write(str(diff_obj[0][i])+' ')
                                        	fp.write(str(diff_obj[-2])+' '+str(diff_obj[-1])+'\n')
	    			with open(filename_train, 'a') as fp:
	    				fp.write(slide_name+'/'+subname+'\n')
	    			cropped.save(os.path.join(output_dir, 'Images', slide_name, subname+file_extension))
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
            img.save(os.path.join(output_dir, 'Images', slide_name, name+file_extension))
