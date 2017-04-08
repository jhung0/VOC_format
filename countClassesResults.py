import numpy as np
from PIL import Image
from IPython.display import display
import pickle
import os
from sys import argv
'''
counts classes for results after testing
python countClassesResults.py [PID number] [test_name]
'''
#get PID and test_name
PID = argv[1]
test_name = argv[2]

image_path = '/home/ubuntu/try1/data/Images/'
#classes = ['__background__', 'rbc', 'other']
classes = ['__background__', 'rbc', 'tro', 'sch', 'ring', 'gam', 'leu']
THRESHOLD = .5 #1.0/len(classes)#0.65
path = '/home/ubuntu/try1/results/'+test_name

#for each class get the number of bounding boxes with probability >= THRESHOLD
for cls in range(1,len(classes)):
        filtered_boxes=0
        #open nms detection file
        filename = PID+'_det_'+test_name+'_'+classes[cls]+'.txt'
        with open(os.path.join(path, filename), 'r') as f:
            for line in f.readlines():
                line_list = line.split()
                #if the detection has probability above the threshold
                if float(line_list[1]) >= THRESHOLD:
                        filtered_boxes += 1
	print classes[cls] + ':' + str(filtered_boxes) + '.'
    
 
