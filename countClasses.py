import os
from sys import argv
'''
Count the number of cells in each class (ground truth)
python countClasses.py [test name: train, trainfull, test]
'''
_data_path ='/home/ubuntu/try1/data'
test_name = argv[1]
classes = ['rbc', 'tro', 'sch', 'ring', 'gam', 'leu', 'difficult']
counts = [0]*len(classes)

with open(os.path.join(_data_path, 'ImageSets', test_name + '.txt')) as fp:
	indices = fp.readlines()

for index in indices:
	filename = os.path.join(_data_path, 'Annotations', index.strip() + '.txt')
        with open(filename) as f:
            data = f.readlines() #each row is an element in a list

	for datum in data:
		info = datum.strip().split(' ')
		cls = info[-2]
		counts[classes.index(cls)] += 1
		if info[-1] == 'True':
			counts[-1] += 1
	'''
	print index
	for i in range(len(classes)):
		print classes[i] +':'+ str(counts[i]) + '.'
	'''
for i in range(len(classes)):
	print classes[i] +':'+ str(counts[i]) + '.'
