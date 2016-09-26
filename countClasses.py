import os

'''
Count the number of cells in each class (ground truth)
'''
_data_path ='/home/ubuntu/try1/data'
test_name = 'trainfull'
classes = ['rbc', 'tro', 'ring', 'sch', 'gam', 'leu']
counts = [0]*len(classes)

with open(os.path.join(_data_path, 'ImageSets', test_name + '.txt')) as fp:
	indices = fp.readlines()

for index in indices:
	filename = os.path.join(_data_path, 'Annotations', index.strip() + '.txt')
        with open(filename) as f:
            data = f.readlines() #each row is an element in a list

	for datum in data:
		cls = datum.strip().split(' ')[-2]
		counts[classes.index(cls)] += 1
	print index
	for i in range(len(classes)):
		print classes[i] +':'+ str(counts[i]) + '.'
