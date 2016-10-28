from PIL import Image
from operator import itemgetter
import pickle
import os
from sys import argv
from sklearn.metrics import confusion_matrix
import numpy as np
import subprocess
'''
Create image and text files to be used for training with caffe
Has option to not include detections classified as rbc
'''

net = argv[1] #net
prototxt = argv[2] #prototxt
DET = argv[3] #PID trainfull
DET_test = argv[4] #PID test
REMOVE_RBC = True
classes_train = ['__background__', 'rbc', 'other']
classes_test = ['__background__', 'rbc', 'tro', 'sch', 'ring', 'gam', 'leu']

THRESHOLD = 1.0/(len(classes_train))
print 'detection threshold ', THRESHOLD
MIN_OVERLAP = 0.5
data_path = '/home/ubuntu/try1/data/'
image_path = os.path.join(data_path, 'Images')
cfg_file = "/home/ubuntu/py-faster-rcnn/experiments/cfgs/faster_rcnn_end2end.yml"
TRAIN_DATA_ROOT= '/home/ubuntu/caffe/examples/try1/train'
VAL_DATA_ROOT='/home/ubuntu/caffe/examples/try1/test'
DATA = '/home/ubuntu/caffe/data/try1'
INCLUDE_RBCs = False

def getDetections(cls, filename, threshold, test_name = 'trainfull', path='/home/ubuntu/try1/results'):
    filtered_detections = []
    #filter detections
    with open(os.path.join(path, test_name, filename), 'r') as f:
            for line in f.readlines():
                line_list = line.split()
                #if the detection has probability above the threshold
                if float(line_list[1]) >= threshold:
                        det_index = line_list[0]
			#if the detection's index matches any index already in filtered detections, add to that dictionary; else, make a new dictionary 
                        if any(d['index'] == det_index for d in filtered_detections):
                                fi = map(itemgetter('index'), filtered_detections).index(det_index)
                                filtered_detections[fi]['boxes'].append([float(i) for i in line_list[2:]])
                                filtered_detections[fi]['classes'].append(cls)
                        else:
                                filtered_detections.append({'boxes': [[float(i) for i in line_list[2:]]], 'classes': [cls], 'index': line_list[0]})
    return filtered_detections

def getGroundTruth(data_path, classes, test_name='trainfull'):
    gt = []
    #get ground truth and filtered detections
    image_set_file = os.path.join(data_path, 'ImageSets', test_name + '.txt')
    with open(image_set_file) as f:
        image_index = [x.strip() for x in f.readlines()]

    npos = [0]*len(classes) #number of gt for each class
    for i, index in enumerate(image_index):
        gt.append(load_try1_annotation(classes, data_path, index))
        gt[i]['index'] = index
        gt[i]['det'] = np.zeros(len(gt[i]['gt_classes']))
        for jj, cls in enumerate(classes):
                cls_indices = np.where(gt[i]['gt_classes'] == classes.index(cls))[0]
                npos[jj] += sum(1-gt[i]['difficult'][cls_indices])
    return gt

def saveTxt(test_name, img_filename, label, data_dir=DATA):
	if test_name == 'trainfull':
		test_name = 'train'
	with open(os.path.join(data_dir, test_name+'.txt'), "a+") as f:
		f.write(img_filename + ' ' + str(label)+'\n')
		
def load_try1_annotation(classes, data_path, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(data_path, 'Annotations', index + '.txt')
        with open(filename) as f:
            data = f.readlines() #each row is an element in a list
            #non_diff_objs = [data[ix] for ix in range(len(data)) if data[ix].strip().split(' ')[-1] != 'True']
            #data = non_diff_objs
        num_objs = len(data)
        if num_objs == 0:
                return None
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros(num_objs)
        difficult = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix in range(num_objs):
            try:
                x1, y1, x2, y2, cls, df = data[ix].strip().split(' ')
            except:
                raise Exception('Error in reading data, line %s:%s'%(str(ix+1), data[ix]))
            #pixel indexes 0-based
	    if cls not in classes:
		continue
	    else:
            	boxes[ix, :] = [x1, y1, x2, y2]
            	gt_classes[ix] = classes.index(cls)
            	difficult[ix] = df == 'True'

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'flipped' : False,
                'difficult' : difficult
                }

#clear existing files
for name in [TRAIN_DATA_ROOT, DATA, VAL_DATA_ROOT]:
	clear_dir = name
    	for f in os.listdir(clear_dir):
		try:
			os.remove(os.path.join(clear_dir, f))
		except:
			for fp in os.listdir(os.path.join(clear_dir, f)):
				os.remove(os.path.join(clear_dir, f, fp))

for test_name in ['trainfull', 'test']:
    print test_name
    if test_name == 'test':
	DET = DET_test
        TRAIN_DATA_ROOT = VAL_DATA_ROOT
    counts = [0]*(1+len(classes_test))

    filtered_detections = []
    if INCLUDE_RBCs:
	start_class_idx = 1
    else:
	start_class_idx = 2
    for cls in classes_train[start_class_idx:]:
	#get detections
	filename = str(DET)+'_det_'+test_name+'_'+cls+'.txt'
	filtered_detections.extend(getDetections(cls, filename, THRESHOLD, test_name))
    gt = getGroundTruth(data_path, classes_test, test_name)
    for gt_i in gt:
	index = gt_i['index']
	extension='.jpg' 
        pil_im =Image.open(os.path.join(image_path, index+extension))
	pil_im = pil_im.copy()
	try:
        	filtered_detections_image = filtered_detections[map(itemgetter('index'), filtered_detections).index(index)]#filter detections for those in the same image as the gt
        except:
		print index, ' has no non RBC detections'
		continue
	#index = index.split('/')
	#index = index[0]+'-'+index[1]
        print index
	#for each detection (of any class), find if there's a matching ground truth which has not yet been matched
        for ii, _boxes in enumerate(filtered_detections_image['boxes']):
                ov_max = -float("inf")
                #detection_class = filtered_detections_image['classes'][ii]
		cropped = pil_im.crop((int(_boxes[0]), int(_boxes[1]), int(_boxes[2]), int(_boxes[3])))
                for ngt, gt_boxes in enumerate(gt_i['boxes']):
                        iw = min(_boxes[2], gt_boxes[2]) - max(_boxes[0], gt_boxes[0]) + 1
                        ih = min(_boxes[3], gt_boxes[3]) - max(_boxes[1], gt_boxes[1]) + 1
                        if iw > 0 and ih > 0:
                                #compute overlap as area of intersection / area of union
                                ua = (_boxes[2] - _boxes[0] + 1)*(_boxes[3] - _boxes[1] + 1) + (gt_boxes[2] - gt_boxes[0] + 1)*(gt_boxes[3] - gt_boxes[1] + 1) - iw*ih
                                ov = iw*ih*1.0/ua
                                if ov > ov_max:
                                        ov_max = ov
                                        ngt_max = ngt
		#balance classes
		try:
			current_class = int(gt_i['gt_classes'][ngt_max])
		except:
			current_class = 0
		if max(counts) == counts[current_class]:
			aug = 1
		elif max(counts)*1.0/4 > counts[current_class]:
			aug = 8
		elif max(counts)*1.0/2 > counts[current_class]:
			aug = 4
		else: 
			aug = 2
		if test_name == 'test':
			aug = 1
		for jj in range(0, aug):
			cropped_ = cropped.copy()
                	if ov_max >= MIN_OVERLAP:
				gt_i['det'][ngt_max] = 1	
				cropped_name = os.path.join(index+'_'+str(ngt_max)+'_'+str(ii)+'_'+str(jj)+extension)
                        	if not gt_i['difficult'][ngt_max]:
					saveTxt(test_name, cropped_name, int(gt_i['gt_classes'][ngt_max]))
					counts[int(gt_i['gt_classes'][ngt_max])] += 1
				elif test_name == 'test':
					saveTxt('test_diff', cropped_name, 7)
					counts[7] += 1
                	else:
				cropped_name = os.path.join(index+'_'+'None'+'_'+str(ii)+'_'+str(jj)+extension)
                        	saveTxt(test_name, cropped_name, 0)
				counts[0] += 1
			for _ in range(int(jj/2)%4):
                        	cropped_ = cropped_.rotate(90)
			if jj%2 == 1:
                                cropped_ = cropped_.transpose(Image.FLIP_LEFT_RIGHT)
                	cropped_.save(os.path.join(TRAIN_DATA_ROOT, cropped_name))
        #add each ground truth to the training set
	if test_name == '': #'trainfull':
	    for jj, gt_boxes in enumerate(gt_i['boxes']):
		img_filename = os.path.join(index+'_'+str(jj)+extension)
		try:
			cropped = pil_im.crop((int(gt_boxes[0]), int(gt_boxes[1]), int(gt_boxes[2]), int(gt_boxes[3])))
			cropped.save(os.path.join(TRAIN_DATA_ROOT, img_filename))
			saveTxt(test_name, img_filename, int(gt_i['gt_classes'][jj]))
		except:
			#print index, gt_boxes
			#raise Exception
			continue
    print 'counts', counts
