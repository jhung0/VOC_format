from glob import glob
import sys 
import fileinput
from PIL import Image
import random
import os
import xml.etree.ElementTree as ET
import operator
import numpy as np
from scipy import stats
from shutil import copyfile
import argparse
'''
Randomly takes X subsamples of full image and outputs in other folder
Usage: python VOC_format_balance2.py --[parse args] inputs
'''
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train and test files for Faster R-CNN')
    parser.add_argument('--train', dest='TRAINING_SET_DIR',
                        help='list of directories in the training set', nargs='+', type=str)
    parser.add_argument('--test', dest='TEST_SET_DIR',
                        help='list of directories in the test set', nargs='+', type=str)
    parser.add_argument('--balance', dest='BALANCE',
                        help='whether to try to balance infected classes', default=True, type=bool)
    parser.add_argument('--difficult', dest='DIFFICULT',
                        help='whether there is a difficult tag', default=True, type=bool)
    parser.add_argument('--rotate', dest='ROTATE',
                        help='whether to rotate by 90, 180, 270 degrees', default=True, type=bool)
    parser.add_argument('--images', dest='images',
                        help='file with names of files to test',
                        default='/home/ubuntu/try1/data/ImageSets/test.txt', type=str)
    parser.add_argument('--classes', dest='classes',
			help='list of class names', default=['rbc', 'tro', 'sch', 'ring', 'gam', 'leu'], nargs='+', type=str)
    parser.add_argument('--limit', dest='RBC_LIMIT',
			help='set to none or to percentage of subimages per image containing only the dominant class', default=0.04,type=float)
    parser.add_argument('--num_subimages', dest='num_subimages', help='number of subimages (not including rotations)', default=100, type=int) 
    parser.add_argument('--subimage_size', dest='subimage_size', help='size of subimages (px)', default=448, type=int)
    parser.add_argument('--output', dest='output_dir',
			help='output directory',default='/home/ubuntu/try1', type=str)
    parser.add_argument('--input', dest='input_dir', help='input directory', default='/var/www/html/LabelMeAnnotationTool/Images' , type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

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
        
    if not args.DIFFICULT and label[0] == 'd':
        label = 'uncertain'
    elif label == 'a':
	label = 'gam'
        difficult = True
    elif label[0] == 'd':
        difficult = True
        label = label[1:]
    #if label is not in the CLASSES list, then skip
    if label not in args.classes:
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
def chooseTrainOrTest(filename, training_dir, test_dir):
    if all(dd in train_dir for dd in test_dir):
    	if random.random() < 0.8:
        	return 'train'
    	return 'test'
    else:
	if any(tt in filename for tt in training_dir):
        	return 'train'
    	return 'test'

#if file exists, remove
def removeIfExists(filename): 
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
	if not os.path.exists(os.path.dirname(filename_annotation)):
		os.makedirs(os.path.dirname(filename_annotation))
	with open(filename_annotation, 'a+') as fp:
		for datum in boxCoords:
			fp.write(str(datum)+' ')
		#print boxCoords, label
                fp.write(str(label)+' '+str(difficult)+'\n')

#save image set
def saveImageSet(filename_train, name):
	with open(filename_train, 'a+') as fp:
		fp.write(name+'\n')

def clearFiles(output_dir, all_slide_names):
	#clear existing files
	for name in ['Annotations', 'ImageSets', 'Images']:
        	for s_name in all_slide_names:
                	if name == 'ImageSets':
                        	clear_dir = os.path.join(output_dir, name)
                	else:
                        	clear_dir = os.path.join(output_dir, name, s_name)
			if os.path.exists(clear_dir):
                		for f in os.listdir(clear_dir):
                        		os.remove(os.path.join(clear_dir, f))

def getXml(filename, file_extension):
	path_collection = os.path.abspath(filename).split('/Images/')
	path_annotations = os.path.join(path_collection[0], 'Annotations')
        xml_name = path_collection[1].split(file_extension)[0]
        for name in [xml_name, xml_name.upper(), xml_name.lower()]:
        	try_filename_xml = os.path.join(path_annotations, name+'.xml')
                if os.path.isfile(try_filename_xml):
                	filename_xml = try_filename_xml
                        break
    	else:
        	#raise Exception('%s xml file not found'%(xml_name))
		print '%s xml file not found'%(xml_name)
		return None
	return filename_xml

def getData(filename_xml, classes, counts):
	data = []
        tree = ET.parse(filename_xml)
        root = tree.getroot()
        for obj in root.findall('object'):
        	try:
                	xmin, ymin, xmax, ymax, label, difficult = extractObjectData(obj)
                except:
                        continue
               	data.append([xmin, ymin, xmax, ymax, label, difficult])
		counts[classes.index(label)] += 1
	return data,counts

def getCrop(width, height, small_size, data):
	#randomly choose top left corner of subimage
        randx = random.randint(0, width-small_size)
        randy = random.randint(0, height-small_size)
        #print('top left corner coordinates:%s,%s'%(randx, randy))
	
        return img.crop((randx, randy, randx+small_size, randy+small_size)), getObjectDataFromCrop(data, randx, randy, small_size)

def saveAll(cropped, data_crop, classes, annotation_name, image_name, imageset_name, imageset_item, small_size, ii):
	#if Annotation file exists, remove
        filename_annotation = removeIfExists(annotation_name)

        #save crop
	for _ in range(ii):
        	cropped = cropped.rotate(90)
        cropped.save(image_name)

        #adjust data
        for object_data in data_crop:
               	adjusted_data = np.array(object_data[0:4]).copy()
                for _ in range(ii):
                	#print ii, adjusted_data
                        #adjusted_data = np.array([small_size - adjusted_data[3], adjusted_data[0], small_size - adjusted_data[1], adjusted_data[2]])
                        adjusted_data = np.array([adjusted_data[1], small_size - adjusted_data[2], adjusted_data[3], small_size - adjusted_data[0]])
                #count
                if object_data[-1] == False:
                	counts[classes.index(object_data[-2])] += 1
                #save annotation and image set
                saveAnnotation(filename_annotation, adjusted_data, object_data[-2], object_data[-1])
        saveImageSet(imageset_name, imageset_item)


	
if __name__ == '__main__':
	args = parse_args()

	output_dir = args.output_dir#os.path.join('/Users', 'jyhung', 'Documents', 'VOC_format', 'data')
	input_dir = args.input_dir
	print 'output directory', output_dir
	print 'input directory', input_dir
	TRAINING_SET_DIR = args.TRAINING_SET_DIR
	num_subimages = args.num_subimages
	print 'number of subimages (not including rotations)', num_subimages
	small_size = args.subimage_size 
	print 'size of subimages (px)', small_size
	train_dir = args.TRAINING_SET_DIR
	test_dir = args.TEST_SET_DIR
	print train_dir, test_dir
	all_slide_names = train_dir + test_dir  
	random.shuffle(all_slide_names)
	print 'slide name', all_slide_names
	RBC_LIMIT = int(args.RBC_LIMIT*num_subimages)
	print 'number of RBC only crops per full image is limited to', RBC_LIMIT

	clearFiles(output_dir, all_slide_names)


	#if ROTATE, 4* the number of subimages
	ROTATE = args.ROTATE
	if ROTATE:
    		num_subimages = 4*num_subimages
    		
	#for each image, subsample image and for each subimage, create associated file with bounding box and class information
	filenum = 1
	classes = args.classes 
	counts = [0]*len(classes)
	for current_dir_ in all_slide_names:
		current_dir = os.path.join(input_dir, current_dir_)
		shuffled_current_dir = os.listdir(current_dir)
		random.shuffle(shuffled_current_dir)
		for filename in shuffled_current_dir:
			filename = os.path.join(current_dir, filename)
			file_, file_extension = os.path.splitext(filename)
			s_name = os.path.split(current_dir)[1]
    			print file_, file_extension, s_name
			img = Image.open(filename)

    			#choose whether file is part of training or testing set
    			train_or_test = chooseTrainOrTest(file_, train_dir, test_dir)
    
    			filenum += 1
    			filename_train = os.path.join(output_dir, 'ImageSets',train_or_test+'.txt')

    			width = img.size[0]
    			height = img.size[1]
    
    			#get associated xml file and parse for all objects
			filename_xml = getXml(filename, file_extension)
			if not filename_xml:
				continue
    			print filename_xml    

			data, counts = getData(filename_xml, classes, counts)
			if len(data) == 0:
				print 'no objects'
				continue
			if train_or_test  == 'train':
				total_num_objects = len(data)
				if ROTATE:
					total_num_objects = 4*total_num_objects
				num_objects = 0
				sub = 0
				num_rbc_only = 0
				while sub < num_subimages and num_objects < 2*total_num_objects:
        				empty = True
					cropped_, data_crop = getCrop(width, height, small_size, data)
		
					#write and save annotation file, only including data that are within the bounds of the subimage
					#only save annotation file if there's at least 1 object that is not difficult
					#proceed if there's an object that has label in minor label list or if the image has a non difficult object (checks if it only contains rbcs and checks if the limit has been reached)
					minor_label_list = getMinorLabels(counts, classes)
					#print minor_label_list
					max_class = classes[counts.index(max(counts))]
					if any(object_data[-2] in minor_label_list for object_data in data_crop):
						empty = False
						for ii in range(0, 4, 4/(4**ROTATE)):
							cropped = cropped_.copy()
                                			sub += 1
                                			subname = os.path.basename(file_) + '_' + str(sub)
						
							annotation_name = os.path.join(output_dir, 'Annotations',s_name, subname+'.txt')
							image_name = os.path.join(output_dir, 'Images', s_name, subname+file_extension)
							imageset_item = s_name+'/'+subname
							saveAll(cropped, data_crop, classes, annotation_name, image_name, filename_train, imageset_item, small_size, ii)
							num_objects += len(data_crop)
					#randomly rotate
					elif len(data_crop) > 0:
						sub += 1
						all_rbc = all(object_data[-2] == max_class for object_data in data_crop)
                                                if all_rbc and num_rbc_only >= RBC_LIMIT:
                                                        continue
                                                elif all_rbc:
                                                        num_rbc_only += 1
                                                empty = False

						cropped = cropped_.copy()
						rand_rotate = random.randint(0,3)
						subname = os.path.basename(file_) + '_' + str(sub)

						annotation_name = os.path.join(output_dir, 'Annotations',s_name, subname+'.txt')
                                               	image_name = os.path.join(output_dir, 'Images', s_name, subname+file_extension)
                                                imageset_item = s_name+'/'+subname
						saveAll(cropped, data_crop, classes, annotation_name, image_name, filename_train, imageset_item, small_size, rand_rotate)
						num_objects += len(data_crop)
				filename_train = os.path.join(output_dir, 'ImageSets', 'trainfull.txt')
			#elif train_or_test == 'test': #full image with all annotations
        		empty = False #True
        		#if Annotation file exists, remove
        		name = os.path.basename(file_)
        		filename_annotation = removeIfExists(os.path.join(output_dir, 'Annotations', s_name, name+'.txt'))
        
        		for object_data in data:
            			empty = False
	    			saveAnnotation(filename_annotation, object_data[0:4], object_data[-2], object_data[-1])
        		if not empty:
	    			saveImageSet(filename_train, s_name+'/'+name)
				image_name = os.path.join(output_dir, 'Images', s_name, name+file_extension)
				if not os.path.exists(os.path.dirname(image_name)):
					os.makedirs(os.path.dirname(image_name))
            			copyfile(filename, image_name)

#print counts
