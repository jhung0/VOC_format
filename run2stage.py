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
from PIL import Image
import skimage
import uuid

'''
runs images through 2 stage model, saves label matrices
python run2stage.py ([--images image_list.txt]) [--cfg1 experiments/cfgs/faster_rcnn_end2end.yml] [--prototxt1 xxx.prototxt] [--model1 xxx.caffemodel] [--prototxt2 deploy.prototxt] [--model2 xxx.caffemodel]
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
                        default='/home/ubuntu/py-faster-rcnn/models/VGG_CNN_M_1024/faster_rcnn_end2end/test.prototxt', type=str)
    parser.add_argument('--prototxt2', dest='prototxt2',
                        help='prototxt file defining the network',
                        default='/home/ubuntu/py-faster-rcnn/caffe-fast-rcnn/models/bvlc_reference_caffenet/deploy.prototxt', type=str)
    parser.add_argument('--model1', dest='caffemodel1',
                        help='model to test',
                        default='/home/ubuntu/py-faster-rcnn/output/faster_rcnn_end2end/train/vgg_cnn_m_1024_faster_rcnn_lr0.01_iter_100000.caffemodel', type=str)
    parser.add_argument('--model2', dest='caffemodel2',
                        help='model to test',
                        default='/home/ubuntu/py-faster-rcnn/caffe-fast-rcnn/models/bvlc_reference_caffenet/caffenet_train_iter_40000.caffemodel', type=str)
    parser.add_argument('--cfg1', dest='cfg_file1',
                        help='optional config file', default='/home/ubuntu/py-faster-rcnn/experiments/cfgs/faster_rcnn_end2end.yml', type=str)
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
    parser.add_argument('--output', dest='output_dir',
			help='output directory',default='/home/ubuntu/svg', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def StageOne(file_, prototxt, model, classes, THRESHOLD=1.0/3, num_images = 1, output_dir = '/home/ubuntu/py-faster-rcnn/output' ):
    '''
	run one image through object detector to classify each cell as background, rbc, or other
	Return: all boxes with score above THRESHOLD
    '''
    net = caffe.Net(prototxt, model, caffe.TEST)
    print 'prototxt ', prototxt
    print 'caffemodel ', model
    net.name = os.path.splitext(os.path.basename(model))[0]
 
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
    print 'Applying NMS to all detections'
    nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)
    with open(det_file, 'wb') as f:
        cPickle.dump(nms_dets, f, cPickle.HIGHEST_PROTOCOL)
    return nms_dets

def StageTwo(file_path, prototxt, model, detections, classes):
    '''
	run detections from one image through image classifier
	Return: all detections 
    '''
    net = caffe.Net(prototxt, model, caffe.TEST)
    print 'prototxt ', prototxt
    print 'caffemodel ', model
    net.name = os.path.splitext(os.path.basename(model))

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.array([189.97, 133.83, 149.26  ]))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255.0)
    transformer.set_channel_swap('data', (2,1,0))

    probs = np.zeros((len(detections), len(classes)))

    full_im = Image.open(file_path)
    #full_im = caffe.io.load_image(file_path)
    full_im = full_im.copy()    #caffe.io.load_image()
    for det_index, det in enumerate(detections):
	print det_index, det
	img = full_im.crop((int(det[0]), int(det[1]), int(det[2]), int(det[3])))
	img.save('/home/ubuntu/stage2.jpg')
	img = caffe.io.load_image('/home/ubuntu/stage2.jpg')
    	net.blobs['data'].reshape(1, 3, 227,227)
    	net.blobs['data'].data[...] = transformer.preprocess('data', np.array(img))#np.array(img))
    	output = net.forward()
	print 'output ', output['prob']    
    	probs[det_index] = output['prob']
    return probs

def WriteXml(root, box, cls, attributes_text, index):
	'''
	write to Element Tree
	'''
	#print 'box ', box
        object_ = ET.Element('object')
        root.append(object_)
        name_ = ET.SubElement(object_, 'name')
        name_.text = cls
        deleted_ = ET.SubElement(object_, 'deleted')
        deleted_.text = "0"
        verified_ = ET.SubElement(object_, 'verified')
        verified_.text = "0"
        occluded_ = ET.SubElement(object_, 'occluded')
        occluded_.text = 'no'
        attributes_ = ET.SubElement(object_, 'attributes')
        attributes_.text = attributes_text
        parts_ = ET.SubElement(object_, 'parts')
        hasparts_ = ET.SubElement(parts_, 'hasparts')
        ispartof_ = ET.SubElement(parts_, 'ispartof')

        date_ = ET.SubElement(object_, 'date')
        date_.text = str(0)
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
                    x_.text = str(box[((i + i%2)%4)])
                    y_ = ET.SubElement(pt_, 'y')
                    y_.text = str(box[(i + (i+1)%2)])

def CreateXml(LabelMe_path, file_, stage1_dets, stage2_probs, classes):
    '''
	create LabelMe xml file from detection coordinates 
    '''
    print LabelMe_path, file_
    LabelMe_annotation_dir = os.path.join(LabelMe_path, 'Annotations')    
    file_ = file_.split('Images/')[1]
    print file_
    image_dir = file_.split('/')[0]
    #make LabelMe xml annotation file
    LabelMe_file = os.path.join(LabelMe_annotation_dir, file_+'.xml')
    print LabelMe_file
    #clear existing annotations
    tree = ET.parse(LabelMe_file)
    root = tree.getroot()
    root.find('folder').text = image_dir
    for obj in root.findall('object'):
            root.remove(obj)
    #get detection coordinates
    rbc_dets = stage1_dets[1][0]
    other_dets = stage1_dets[2][0]
    print rbc_dets[0]
    #for each set of coordinates, create object instance
    for index, box in enumerate(rbc_dets):
	box = rbc_dets[index][:4]
	#print box
	attributes = str(rbc_dets[index][-1])
	writeXML(root, box, classes[1], attributes, index)
    for index_other, box in enumerate(other_dets):
	index += 1
	box = other_dets[index_other][:4]
	attributes = str(other_dets[index_other][-1])
	print stage2_probs[index_other], np.argmax(stage2_probs[index_other])
	writeXML(root, box, classes[np.argmax(stage2_probs[index_other])], attributes, index)

    tree.write(LabelMe_file)
    os.chmod(LabelMe_file, 0o777)


def WriteRect(box, cls, score):
    '''
    write rect to Element Tree
    '''
    #print 'box ', box
    rect_ = ET.Element('rect')
    rect_.set('description', cls)
    rect_.set('score', str(score))
    rect_.set('x', str(int(box[0])))
    rect_.set('y', str(int(box[1])))
    rect_.set('width', str(int(box[2]-box[0])))
    rect_.set('height', str(int(box[3]-box[1])))
    return rect_

def WriteImage(file_):
    '''
    write image
    '''
    image_ = ET.Element('image')
    image_.set('id', str(uuid.uuid3(uuid.NAMESPACE_DNS, file_)))
    return image_

def CreateSvg(output_dir, file_, detections, probs, classes):
    '''
	create svg using original image, detections, probability distributions and save to output
    '''
    output = os.path.join(output_dir, os.path.basename(file_.rsplit(".",1)[0]) + '.svg')

    try:
    	#clear existing annotations
    	tree = ET.parse(output)
    	root = tree.getroot()
    	for obj in root.findall('rect'):
            root.remove(obj)
    except:
	root = ET.Element('svg')
	tree = ET.ElementTree(root)
    #get detection coordinates
    rbc_dets = detections[1][0]
    other_dets = detections[2][0]
    
    image_ = WriteImage(file_)
    root.append(image_)

    #for each set of coordinates, create object instance
    for index, box in enumerate(rbc_dets):
        box = rbc_dets[index]
        #print box
        attributes = str(rbc_dets[index][-1])
        root.append(WriteRect(box[:4], classes[1], box[4]))
    for index_other, box in enumerate(other_dets):
        index += 1
        box = other_dets[index_other]
        attributes = str(other_dets[index_other][-1])
        print stage2_probs[index_other], np.argmax(stage2_probs[index_other])
        root.append(WriteRect(box[:4], classes[np.argmax(stage2_probs[index_other])], box[4]))

    tree.write(output)
    os.chmod(output, 0o777)
    return 0

def get_files(ImageSet_test):
    test_files = []
    with open(ImageSet_test) as f:
        for file_ in f.readlines():
            test_files.append(file_.strip())
    return test_files

def get_dimensions(file_):
    with Image.open(file_) as im:
	return im.size

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file1 is not None:
        cfg_from_file(args.cfg_file1)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.GPU_ID = args.gpu_id

    #print('Using config:')
    #pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel1) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel1))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    
    #get test image filenames
    imageSet_test = args.images
    test_files = get_files(imageSet_test)#['/home/ubuntu/try1/data/Images/g8_t1_up/g6010001.jpg']

    classes1 = args.classes1
    classes2 = args.classes2

    #for each image in the list, run through stage 1, then run through stage 2, then convert results and create xml file
    for file_index, file_ in enumerate(test_files):
	dimensions = get_dimensions(file_)
	cfg_from_list(['TEST.SCALES', str([min(dimensions)]), 'TEST.MAX_SIZE', str(max(dimensions))])
	pprint.pprint(cfg)
	nms_dets = StageOne(file_, args.prototxt1, args.caffemodel1, classes1, THRESHOLD=1.0/len(classes1), output_dir=args.output_dir)
	stage2_probs = StageTwo(file_, args.prototxt2, args.caffemodel2, nms_dets[classes1.index('other')][0], classes2)
	#print 'stage 2', stage2_dets
	CreateSvg(args.output_dir, file_, nms_dets, stage2_probs, classes2)







