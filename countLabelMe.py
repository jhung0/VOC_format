import numpy as np
from PIL import Image
from IPython.display import display
import pickle
import os
from sys import argv
import xml.etree.ElementTree as ET
'''
counts classes for results after testing
python countLabelMe.py [path to LabelMe Annotations]
'''
#get path to LabelMe Annotations
path = argv[1]

#for each file in the directory, count classes
counts = []
labels = []

for file_ in os.listdir(path):
    file_ = os.path.join(path, file_)
    tree = ET.parse(file_)
    root = tree.getroot()
    for obj in root.findall('object'):
	deleted = int(obj.find('deleted').text)
	if deleted:
		continue
    	label = obj.find('name').text
	
	if label in labels:
		counts[labels.index(label)] += 1
	else:
		counts.append(1)
		labels.append(label)	
print labels
print counts 
 
