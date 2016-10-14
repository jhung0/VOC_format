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
from shutil import copyfile
'''
Randomly takes X subsamples of full image and outputs in other folder
Usage: python VOC_format.py [Output directory] [Image files]
Includes flags for different options. 
'''
BALANCE = True #whether to try to balance infected classes
DIFFICULT = True #whether there's a difficult tag
ROTATE = True #whether to (in addition to original subimages), flip and rotate by 90, 180, 270 
UNCERTAIN_CLASS = False #don't have uncertain class, either ignore or tag as difficult
FLIP = False  #whether to flip
TRAINING_SET_DIR = ['g16_t1_up', 'g16_t2_up', 'g12', 'g11_t1', 'g7_t1_up']
CLASSES = ['rbc', 'tro', 'sch', 'ring', 'gam', 'leu']
#extract data from xml file
def extractObjectData(obj):
    deleted = int(obj.find('deleted').text)
    label = obj.find('name').text.strip()
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
    #if label is not in the CLASSES list, then skip
    if label not in CLASSES:
	print label
	raise Exception
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
def chooseTrainOrTest(rr, filename, all_slide_names):
    if len(all_slide_names) == 1:
    	if rr < 0.8:
        	return 'train'
    	return 'test'
    else:
	if any(tt in filename for tt in TRAINING_SET_DIR):
        	return 'train'
    	return 'test'

#if file exists, remove
def removeIfExists(output_dir, name): 
    filename = os.path.join(output_dir, name)
    try:
        os.remove(filename)
    except OSError:
        pass
    return filename

#output objects contained in a crop (coordinates adjusted)
def getObjectDataFromCrop(data, randx, randy, small_size):
	data_crop = []
	for object_data in data:
		adjusted_data = np.array(object_data[0:4]).copy()
		adjusted_data = adjusted_data - np.array([randx, randy, randx, randy])
		if np.all(adjusted_data >= 0) and np.all(adjusted_data < small_size):
			data_crop.append(adjusted_data.tolist()+[object_data[-2], object_data[-1]])
	return data_crop

#output list of classes sorted by count
#if the top 2 classes are more than 4 times the number of the lesser classes, then the top 2 are not minor, otherwise only the top 1 is not minor
def getMinorLabels(counts, classes):
	counts_infected_ind = [x[0] for x in sorted(enumerate(counts), key=lambda x:x[1])] #ascending order
	if counts[counts_infected_ind[-3]] > 0 and counts[counts_infected_ind[-2]]*1.0/counts[counts_infected_ind[-3]]:	
		counts_infected_ind = counts_infected_ind[:-2]
	else:
		counts_infected_ind = counts_infected_ind[:-1]	
	return [classes[i] for i in counts_infected_ind]

#save annotated coordinates, label, difficult status to file
def saveAnnotation(filename_annotation, boxCoords, label, difficult):
	with open(filename_annotation, 'a') as fp:
		for datum in boxCoords:
			fp.write(str(datum)+' ')
		#print boxCoords, label
                fp.write(str(label)+' '+str(difficult)+'\n')

#save image set
def saveImageSet(filename_train, name):
	with open(filename_train, 'a') as fp:
		fp.write(name+'\n')

output_dir = argv[1]#os.path.join('/Users', 'jyhung', 'Documents', 'VOC_format', 'data')
input_dir = argv[2]
other_dir = argv[3:]
print 'output directory', output_dir
print 'input directory', input_dir
print 'training set directories', TRAINING_SET_DIR
num_subimages = 100
print 'number of subimages (not including rotations)', num_subimages
small_size = 448 
print 'size of subimages (px)', small_size
slide_name = os.path.split(input_dir)[1] #os.path.basename(os.path.dirname(input_dir))
other_slide_names = [os.path.split(other_dir[i])[1] for i in range(len(other_dir))]
all_slide_names = other_slide_names + [slide_name]
random.shuffle(all_slide_names)
print 'slide name', slide_name, other_slide_names, all_slide_names
RBC_LIMIT = int(0.05*num_subimages)
print 'number of RBC only crops per full image is limited to', RBC_LIMIT

#clear existing files
for name in ['Annotations', 'ImageSets', 'Images']:
	for s_name in all_slide_names:
    		if name == 'ImageSets':
			clear_dir = os.path.join(output_dir, name)
    		else:
			clear_dir = os.path.join(output_dir, name, s_name)
    		for f in os.listdir(clear_dir):
			os.remove(os.path.join(clear_dir, f))

#if train.txt or test.txt exists, remove
#for train_or_test in ['train', 'test']:
#    filename_train = removeIfExists(output_dir, 'ImageSets', train_or_test+'.txt')

#if FROTATE, 8* the number of subimages
if ROTATE:
    num_subimages = 4*num_subimages
if FLIP:
    num_subimages = 2*num_subimages
    		
#for each image, subsample image and for each subimage, create associated file with bounding box and class information
filenum = 1
classes = ['rbc', 'tro', 'sch', 'ring', 'gam', 'leu']
counts = [0]*len(classes)
for current_dir_ in all_slide_names:
  current_dir = os.path.join(os.path.split(input_dir)[0], current_dir_)
  shuffled_current_dir = os.listdir(current_dir)
  random.shuffle(shuffled_current_dir)
  for filename in shuffled_current_dir:
    filename = os.path.join(current_dir, filename)
    file_, file_extension = os.path.splitext(filename)
    s_name = os.path.split(current_dir)[1]
    print file_, file_extension, s_name
    img = Image.open(filename)

    #choose whether file is part of training or testing set
    train_or_test = chooseTrainOrTest(random.random(), file_, all_slide_names)
    
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
	counts[classes.index(label)] += 1
    if train_or_test  == 'train':
	total_num_objects = len(data)
	if ROTATE:
		total_num_objects = 4*total_num_objects
	if FLIP:
		total_num_objects = 2*total_num_objects
	num_objects = 0
	sub = 0
	num_rbc_only = 0
	while sub < num_subimages and num_objects < 2*total_num_objects:
        	empty = True
		#randomly choose top left corner of subimage
		randx = random.randint(0, width-small_size)
		randy = random.randint(0, height-small_size)
		#print('top left corner coordinates:%s,%s'%(randx, randy))
		
		#save cropped image
		cropped_ = img.crop((randx, randy, randx+small_size, randy+small_size))
		
		#write and save annotation file, only including data that are within the bounds of the subimage
		#only save annotation file if there's at least 1 object that is not difficult
		#adjusted_diff = [] #list of difficult objects in image
		
		#get objects from crop
		data_crop = getObjectDataFromCrop(data, randx, randy, small_size)
		#proceed if there's an object that has label in minor label list or if the image has a non difficult object (checks if it only contains rbcs and checks if the limit has been reached)
		minor_label_list = getMinorLabels(counts, classes)
		if any(object_data[-2] in minor_label_list and object_data[-1] == False for object_data in data_crop):
			empty = False
			#frotate image
			for ii in range(0, 8, 8/((4**ROTATE)*(2**FLIP))):
				cropped = cropped_.copy()
				sub += 1
				subname = os.path.basename(file_) + '_' + str(sub)
				#if Annotation file exists, remove
		                filename_annotation = removeIfExists(output_dir, os.path.join('Annotations',s_name, subname+'.txt'))

				if ROTATE:
				    for _ in range(int(ii/2)%4):
                                        cropped = cropped.rotate(90)
                                if FLIP and int(ii/4)%2 == 1:
                                        cropped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
				#save crop
				cropped.save(os.path.join(output_dir, 'Images', s_name, subname+file_extension))
				
				#adjust data
				for object_data in data_crop:
					num_objects += 1
					adjusted_data = np.array(object_data[0:4]).copy()
					if ROTATE:
					    for _ in range(int(ii/2)%4):
						#print ii, adjusted_data
						#adjusted_data = np.array([small_size - adjusted_data[3], adjusted_data[0], small_size - adjusted_data[1], adjusted_data[2]])
                                        	adjusted_data = np.array([adjusted_data[1], small_size - adjusted_data[2], adjusted_data[3], small_size - adjusted_data[0]])
					if FLIP and int(ii/4)%2 == 1:
						adjusted_data = np.array([small_size - adjusted_data[2], adjusted_data[1], small_size - adjusted_data[0],  adjusted_data[3]])
					#count
					if object_data[-1] == False:
						counts[classes.index(object_data[-2])] += 1
					#save annotation and image set
					saveAnnotation(filename_annotation, adjusted_data, object_data[-2], object_data[-1])
				saveImageSet(filename_train, s_name+'/'+subname)
		elif len(data_crop) > 0 and all(object_data[-1] == False for object_data in data_crop):
			sub += 1
			all_rbc = all(object_data[-2] == 'rbc' for object_data in data_crop)
			if all_rbc and num_rbc_only >= RBC_LIMIT:
                                continue
			elif all_rbc:
				num_rbc_only += 1
			empty = False
			subname = os.path.basename(file_) + '_' + str(sub)
			#if Annotation file exists, remove
	                filename_annotation = removeIfExists(output_dir, os.path.join('Annotations',s_name, subname+'.txt'))

			#save crop
			cropped_.save(os.path.join(output_dir, 'Images', s_name, subname+file_extension))

			#save data
			for object_data in data_crop:
				num_objects += 1
				if object_data[-1] == False:
					counts[classes.index(object_data[-2])] += 1
				saveAnnotation(filename_annotation, object_data[0:4], object_data[-2], object_data[-1])
			saveImageSet(filename_train, s_name+'/'+subname)
    elif train_or_test == 'test': #full image with all annotations
        empty = True
        #if Annotation file exists, remove
        name = os.path.basename(file_)
        filename_annotation = removeIfExists(output_dir, os.path.join('Annotations', s_name, name+'.txt'))
        
        for object_data in data:
            empty = False
	    saveAnnotation(filename_annotation, object_data[0:4], object_data[-2], object_data[-1])
        if not empty:
	    saveImageSet(filename_train, s_name+'/'+name)
            copyfile(filename, os.path.join(output_dir, 'Images', s_name, name+file_extension))
