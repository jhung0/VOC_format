import pickle
import os
from sys import argv
from sklearn.metrics import confusion_matrix
'''
display results from pkl files
'''
DET = argv[1]
test_name = argv[2]
iterations = argv[3]
THRESH = 0.5
print 'detection threshold ', THRESH 
MIN_OVERLAP = 0.5
classes = ['__background__', 'rbc', 'tro', 'sch', 'ring', 'gam', 'leu']
base_dir = os.path.join('/home/ubuntu/py-faster-rcnn/output/faster_rcnn_end2end/', test_name)
base_dir = os.path.join(base_dir, 'vgg_cnn_m_1024_faster_rcnn_iter_'+iterations)
data_path = '/home/ubuntu/try1/data/'
path = '/home/ubuntu/try1/results/'
gt = []
filtered_detections = []
def _load_try1_annotation(data_path, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(data_path, 'Annotations', index + '.txt')
        with open(filename) as f:
            data = f.readlines() #each row is an element in a list
        if not self.config['use_diff']:
            non_diff_objs = [data[ix] for ix in range(len(data)) if data[ix].strip().split(' ')[-1] != 'True']
            data = non_diff_objs
        num_objs = len(data)
        if num_objs == 0:
                return None
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        difficult = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix in range(num_objs):
            try:
                x1, y1, x2, y2, cls, df = data[ix].strip().split(' ')
            except:
                raise Exception('Error in reading data, line %s:%s'%(str(ix+1), data[ix]))
            #pixel indexes 0-based
            cls = self._class_to_ind[cls]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
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
        index = next(idx for idx, value in enumerate(f['thresh']) if value<THRESH)
    except:
        index = len(f['thresh']) - 1

    print cls#, index
    print 'sensitivity', f['rec'][index]
    print 'specificity', f['spec'][index]
    print 'precision', f['prec'][index]
    print 'accuracy', (f['tp'][index]+f['tn'][index])*1.0/(f['tp'][index]+f['tn'][index]+f['fn'][index]+f['fp'][index])

    image_set_file = os.path.join(data_path, 'ImageSets', test_name + '.txt')
    with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]

    #extract ground truth   
    for i, index in enumerate(image_index):
            gt.append(load_try1_annotation(data_path, index))
            #filter out objects not in class
            cls_indices = np.where(gt[i]['gt_classes'] == self.classes.index(cls))[0]
            for key in gt[i]:
                try:
                    gt[i][key] = gt[i][key][cls_indices]
                except:
                    pass
            gt[i]['index'] = index
            gt[i]['det'] = np.zeros(len(gt[i]['gt_classes']))

    #get detections
    filename = str(DET)+'_det_'+test_name+'_'+classes[cls]+'.txt'
    
    #filter detections
    with open(os.path.join(path, filename), 'r') as f:
            for line in f.readlines():
                line_list = line.split()
                #if the detection has probability above the threshold
                if float(line_list[1]) >= THRESHOLD:
                        filtered_detections.append({'bbox': [float(i) for i in line_list[2:]], 'class': cls, 'index': line_list[0]})

#make confusion matrix
y_true = []
y_pred = []
for cls in classes[1:]:
	detection_class = 0 #number of detections where the class matches the ground truth class
	for gt_i in gt:
		index = map(itemgetter('index'), gt_i)
		detection_found = 0 #detections associated with this ground truth
		filtered_detections_image = filtered_detections[map(itemgetter('index'), gt).index(index)]#filter detections for those in the same image as the gt
		for detection in filtered_detections_image:
			iw = min(_boxes[2], gt_boxes[2]) - max(_boxes[0], gt_boxes[0]) + 1
                	ih = min(_boxes[3], gt_boxes[3]) - max(_boxes[1], gt_boxes[1]) + 1
                	if iw > 0 and ih > 0:
                    		#compute overlap as area of intersection / area of union
                    		ua = (_boxes[2] - _boxes[0] + 1)*(_boxes[3] - _boxes[1] + 1) + (gt_boxes[2] - gt_boxes[0] + 1)*(gt_boxes[3] - gt_boxes[1] + 1) - iw*ih
                    		ov = iw*ih*1.0/ua
			if ov > MIN_OVERLAP:
				detection_found += 1
				y_pred.append(detection['class'])
				y_true.append(cls)
				if cls == detection['class']:
					detection_class += 1
		if detection_found > 0:
			y_pred.append(classes[0])
			y_true.append(cls)
	for i in range(len([1 for x in filtered_detections if x['class'] == cls]) - detection_class):
		y_pred.append(cls)
		y_true.append(classes[0])

print confusion_matrix(y_true, y_pred, labels=classes)	
