import sys
import os 

def add_path(path):
    #if path not in sys.path:
        sys.path.insert(0, path)

this_dir = '/home/ubuntu/py-faster-rcnn/' #osp.dirname(__file__)
caffe_path = os.path.join(this_dir,  'caffe-fast-rcnn', 'python')
add_path(caffe_path)
lib_path = os.path.join(this_dir, 'lib')
add_path(lib_path)
from utils.timer import Timer
import cv2
import xml.etree.ElementTree as ET
from sys import argv
import argparse
import matplotlib 
matplotlib.use('Agg')
import caffe
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect, apply_nms
import pprint
import numpy as np
import heapq
import cPickle

'''
converts detections from 2 stage model to LabelMe format so that results can be viewed on LabelMe
python convertToLabelMe_2stage.py ([--images image_list.txt]) [--cfg1 experiments/cfgs/faster_rcnn_end2end.yml] [--prototxt1 xxx.prototxt] [--model1 xxx.caffemodel] [--prototxt2 deploy.prototxt] [--model2 xxx.caffemodel]
'''

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--prototxt1', dest='prototxt1',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--prototxt2', dest='prototxt2',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--model1', dest='caffemodel1',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--model2', dest='caffemodel2',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg1', dest='cfg_file1',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--images', dest='images',
                        help='file with names of files to test',
                        default='/home/ubuntu/try1/data/ImageSets/test.txt', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--classes1', dest='classes1',
			help='list of class names for stage 1', default=['__background__', 'rbc', 'other'])
    parser.add_argument('--classes2', dest='classes2',
			help='list of class names', default=['__background__', 'rbc', 'tro', 'sch', 'ring', 'gam', 'leu'],
			type=list) 
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def stage_one(file_, net, classes, THRESHOLD=1.0/3, num_images = 1, output_dir = '/home/ubuntu/py-faster-rcnn/output' ):
    '''
	run one image through object detector to classify each cell as background, rbc, or other
	Return: all boxes with score above THRESHOLD
    '''
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    num_classes = len(classes)
    top_scores = [[] for _ in xrange(num_classes)]
    # all detections are collected into:
    #    all_boxes[cls] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(num_classes)]

    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            raise Exception("HAS_RPN is False")
        print 'image path at', file_
        im = cv2.imread(file_)
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, box_proposals)
        _t['im_detect'].toc()

        _t['misc'].tic()
        for j in xrange(1, num_classes):
            inds = np.where(scores[:, j] > THRESHOLD)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            top_inds = np.argsort(-cls_scores)
            cls_scores = cls_scores[top_inds]
            cls_boxes = cls_boxes[top_inds, :]
            # push new scores onto the minheap
            for val in cls_scores:
                heapq.heappush(top_scores[j], val)
            # if we've collected more than the max number of detection,
            # then pop items off the minheap and update the class threshold
            #if len(top_scores[j]) > max_per_set:
            #    while len(top_scores[j]) > max_per_set:
            #        heapq.heappop(top_scores[j])
            #    thresh[j] = top_scores[j][0]

            all_boxes[j][i] = \
                    np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)

    _t['misc'].toc()

    print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    #only keep boxes with scores above the threshold
    for j in xrange(1, num_classes):
	for i in xrange(num_images):
            inds = np.where(all_boxes[j][i][:, -1] > THRESHOLD)[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
    #print len(all_boxes[0][0]), len(all_boxes[0][1]), len(all_boxes[1][0]), len(all_boxes[1][1]), len(all_boxes[2][0]), len(all_boxes[2][1])
    print 'Applying NMS to all detections'
    nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)
    with open(det_file, 'wb') as f:
        cPickle.dump(nms_dets, f, cPickle.HIGHEST_PROTOCOL)
    return nms_dets

def stage_two(dets, net, classes):
    '''
	run detections from one image through image classifier
	Return: all detections 
    '''
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.array([193.30, 135.15, 153.60  ]))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255.0)
    transformer.set_channel_swap('data', (2,1,0))

    
    img = caffe.io.load_image()
    net.blobs['data'].reshape(1, 3, 227,227)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    output = net.forward()
    
    predicted_class = np.array([np.argmax(output['prob'])])
    return 0

def createXML(LabelMe_path, file_):
    '''
	create LabelMe xml file from detection coordinates 
    '''
    LabelMe_annotation_dir = os.path.join(LabelMe_path, 'Annotations')    
    image_dir = file_.split('/')[0]
    #make LabelMe xml annotation file
    LabelMe_file = os.path.join(LabelMe_annotation_dir, file_+'.xml')
    #print LabelMe_file
    #clear existing annotations
    tree = ET.parse(LabelMe_file)
    root = tree.getroot()
    root.find('folder').text = image_dir
    for obj in root.findall('object'):
            root.remove(obj)
    #get detection coordinates
    for cls in range(1, len(classes)):
        filtered_boxes = []
        #detection_file = str(DET) + '_det_'+test_name+'_'+classes[cls]+'.txt'
        with open(os.path.join(path, detection_file), 'r') as f:
            for line in f.readlines():
                line_list = line.split()
                #if file name matches test file
                if line_list[0].lower() == file_.lower():
                    #if the detection has probability above the threshold
                    if float(line_list[1]) >= THRESHOLD:
                        print line_list
                        filtered_boxes.append([float(i) for i in line_list[1:]])
        #for each set of coordinates, create object instance
        for index, box in enumerate(filtered_boxes):
                object_ = ET.Element('object')
                root.append(object_)
                name_ = ET.SubElement(object_, 'name')
                name_.text = classes[cls]
                deleted_ = ET.SubElement(object_, 'deleted')
                deleted_.text = "0"
                verified_ = ET.SubElement(object_, 'verified')
                verified_.text = "0"
                occluded_ = ET.SubElement(object_, 'occluded')
                occluded_.text = 'no'
                attributes_ = ET.SubElement(object_, 'attributes')
                attributes_.text = str(box[0])
                parts_ = ET.SubElement(object_, 'parts')
                hasparts_ = ET.SubElement(parts_, 'hasparts')
                ispartof_ = ET.SubElement(parts_, 'ispartof')

                date_ = ET.SubElement(object_, 'date')
                date_.text = str(DET)
                id_ = ET.SubElement(object_, 'id')
                id_.text = str(index)
                type_ = ET.SubElement(object_, 'type')
                type_.text = 'bounding_box'
                polygon_ = ET.SubElement(object_, 'polygon')
                username_ = ET.SubElement(polygon_, 'username')
                username_.text = 'anonymous'
                for i in range(4):
                    pt_ = ET.SubElement(polygon_, 'pt')
                    x_ = ET.SubElement(pt_, 'x')
                    x_.text = str(box[1+((i + i%2)%4)])
                    y_ = ET.SubElement(pt_, 'y')
                    y_.text = str(box[1+(i + (i+1)%2)])
        tree.write(LabelMe_file)
    os.chmod(LabelMe_file, 0o777)

def get_files(ImageSet_test):
    test_files = []
    with open(ImageSet_test) as f:
        for file_ in f.readlines():
            test_files.append(os.path.join(imageSet_test.split("ImageSet")[0], "Images", file_.strip()))
    return test_files


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file1 is not None:
        cfg_from_file(args.cfg_file1)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel1) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel1))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net1 = caffe.Net(args.prototxt1, args.caffemodel1, caffe.TEST)
    print 'prototxt ', args.prototxt1
    print 'caffemodel ', args.caffemodel1
    net1.name = os.path.splitext(os.path.basename(args.caffemodel1))[0]
    
    net2 = caffe.Net(args.prototxt2, args.caffemodel2, caffe.TEST)
    print 'prototxt ', args.prototxt2
    print 'caffemodel ', args.caffemodel2
    net2.name = os.path.splitext(os.path.basename(args.caffemodel2))
 
    #get test image filenames
    imageSet_test = args.images
    test_files = get_files(imageSet_test)
    base_path = imageSet_test.split("ImageSet")[0]

    LabelMe_path = '/var/www/html/LabelMeAnnotationTool'
    file_ext = ".jpg"
    classes1 = args.classes1
    classes2 = args.classes2

    #for each image in the list, run through stage 1, then run through stage 2, then convert results and create xml file
    for file_index, file_ in enumerate(test_files):
	nms_dets = stage_one(os.path.join(base_path, file_+file_ext), net1, classes1, THRESHOLD=1.0/len(classes1), output_dir='/home/ubuntu/py-faster-rcnn/output')
	print nms_dets[classes1.index('other')][0]	
	stage2_dets = stage_two(nms_dets[classes1.index('other')][0], net2)
	#createXML(LabelMe_path, file_, nms_dets, stage2_dets)









