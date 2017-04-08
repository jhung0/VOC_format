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
times = ['0', '4hr', '8hr', '12hr', '20hr', '28hr', '36hr', '44hr']
counts = [[] for x in times]
labels = [[] for x in times]

for file_ in os.listdir(path):
    file_ = os.path.join(path, file_)
    for i in range(len(times)):
	#print times[i], os.path.basename(file_)[:len(times[i])]
	if times[i] == os.path.basename(file_)[:len(times[i])]:
		index = i
		break
    else:
	raise Exception(file_) 
    tree = ET.parse(file_)
    root = tree.getroot()
    for obj in root.findall('object'):
	deleted = int(obj.find('deleted').text)
	if deleted:
		continue
    	label = obj.find('name').text
	
	if label in labels[index]:
		counts[index][labels[index].index(label)] += 1
	else:
		counts[index].append(1)
		labels[index].append(label)	
for i in range(len(times)):
	print times[i]
	for j in range(len(labels[i])):
		print labels[i][j], counts[i][j]
 
