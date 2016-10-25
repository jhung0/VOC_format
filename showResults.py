from operator import itemgetter
import pickle
import os
from sys import argv
from sklearn.metrics import confusion_matrix
import numpy as np
'''
display results from pkl files
'''
DET = argv[1]
test_name = argv[2]
iterations = argv[3]
learning_rate = argv[4]
print 'detection threshold ', THRESHOLD 
MIN_OVERLAP = 0.5
#classes = ['__background__', 'rbc', 'tro', 'sch', 'ring', 'gam', 'leu']
classes = ['__background__', 'rbc', 'other']
threshold = 1.0/len(classes)#0.65

base_dir = os.path.join('/home/ubuntu/py-faster-rcnn/output/faster_rcnn_end2end/', test_name)
base_dir = os.path.join(base_dir, 'vgg_cnn_m_1024_faster_rcnn_lr' + str(learning_rate) + '_iter_'+iterations)#os.path.join(base_dir, 'vgg_cnn_m_1024_faster_rcnn_iter_'+iterations+'_lr'+learning_rate)
data_path = '/home/ubuntu/try1/data/'
path = '/home/ubuntu/try1/results/'
gt = []
filtered_detections = []
def load_try1_annotation(classes, data_path, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(data_path, 'Annotations', index + '.txt')
        with open(filename) as f:
            data = f.readlines() #each row is an element in a list
            non_diff_objs = [data[ix] for ix in range(len(data)) if data[ix].strip().split(' ')[-1] != 'True']
            data = non_diff_objs
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
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = classes.index(cls)
            difficult[ix] = df == 'True'

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'flipped' : False,
                'difficult' : difficult
                }

for cls in classes[1:]:
    f = pickle.load(open(os.path.join(base_dir, str(DET)+'_det_'+cls+'_r_p_ap.pkl')))
    #print [idx for idx, value in enumerate(f['thresh']) if value<THRESH]
    try: 
        index = next(idx for idx, value in enumerate(f['thresh']) if value<THRESHOLD)
    except:
        index = len(f['thresh']) - 1

    print cls#, index
    print 'sensitivity', f['rec'][index]
    print 'specificity', f['spec'][index]
    print 'precision', f['prec'][index]
    print 'accuracy', (f['tp'][index]+f['tn'][index])*1.0/(f['tp'][index]+f['tn'][index]+f['fn'][index]+f['fp'][index])

    #get detections
    filename = str(DET)+'_det_'+test_name+'_'+cls+'.txt'
    
    #filter detections
    with open(os.path.join(path, test_name, filename), 'r') as f:
            for line in f.readlines():
                line_list = line.split()
                #if the detection has probability above the threshold
                if float(line_list[1]) >= THRESHOLD:
			det_index = line_list[0]
			if any(d['index'] == det_index for d in filtered_detections):
				fi = map(itemgetter('index'), filtered_detections).index(det_index)
				filtered_detections[fi]['boxes'].append([float(i) for i in line_list[2:]])
				filtered_detections[fi]['classes'].append(cls)
				
			else:
                        	filtered_detections.append({'boxes': [[float(i) for i in line_list[2:]]], 'classes': [cls], 'index': line_list[0]})

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

#make confusion matrix
y_true = []
y_pred = []
#for cls in classes[1:]:
num_detection_class = 0 #number of detections where the class matches the ground truth class
for gt_i in gt:
	    #filter out objects not in class
	    #cls_indices = np.where(gt_i['gt_classes'] == classes.index(cls))[0]
	    #for key in gt_i:
	    #	try:
	    #		    gt_i[key] = gt_i[key][cls_indices]
	    #	except:
	    #	    pass
	    num_detection_found = 0 #number of detections found in an image
	    index = gt_i['index']
	    filtered_detections_image = filtered_detections[map(itemgetter('index'), filtered_detections).index(index)]#filter detections for those in the same image as the gt
	    #for each detection (of any class), find if there's a matching ground truth which has not yet been matched
	    for ii, _boxes in enumerate(filtered_detections_image['boxes']):
		ov_max = -float("inf")
		detection_class = filtered_detections_image['classes'][ii]
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
		if ov_max >= MIN_OVERLAP:
			if not gt_i['difficult'][ngt_max]:#multiple matches to the same gt means multiple entries in the confusion matrix  
				num_detection_found += 1
				y_pred.append(detection_class)
				y_true.append(classes[int(gt_i['gt_classes'][ngt_max])])
				#print y_pred
				#print y_true
				gt_i['det'][ngt_max] = 1
				if gt_i['gt_classes'][ngt_max] == detection_class:
					num_detection_class += 1
		else:
			y_pred.append(detection_class)
			y_true.append(classes[0])	
	    #case for when there is a ground truth but no detection
	    for i in range(len(gt_i['det'])):
		if gt_i['det'][i] == 0:
	   		y_pred.append(classes[0])
			y_true.append(classes[int(gt_i['gt_classes'][i])])
	#print 'y pred', y_pred
	#print 'y true', y_true
print confusion_matrix(y_true, y_pred, labels=classes)	
