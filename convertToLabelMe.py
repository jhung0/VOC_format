import xml.etree.ElementTree as ET
import os

#get test image filenames
test_files = []
data_path = '/home/ubuntu/try1'
ImageSet_test = os.path.join(data_path, 'data', 'ImageSets', 'test.txt') #path of test set file
with open(ImageSet_test) as f:
    for file_ in f.readlines():
        test_files.append(file_.strip())

LabelMe_path = '/var/www/html/LabelMeAnnotationTool'
LabelMe_annotation_dir = os.path.join(LabelMe_path, 'Annotations')
DET = 2943
#create LabelMe xml file from detection coordinates
classes = ['__background__', 'cell']
THRESHOLD = 0.5
path = os.path.join(data_path,'results', 'test')
for file_index, file_ in enumerate(test_files):
    #get detection coordinates
    for cls in range(1,len(classes)):
        filtered_boxes=[]

        detection_file = str(DET) + '_det_test_'+classes[cls]+'.txt'
        with open(os.path.join(path, detection_file), 'r') as f:
            for line in f.readlines():
                line_list = line.split()
                #if file name matches test file
                if line_list[0].lower() == file_.lower():
                    #if the detection has probability above the threshold
                    if float(line_list[1]) >= THRESHOLD:
                        print line_list
                        filtered_boxes.append([float(i) for i in line_list[2:]])

    #make LabelMe xml annotation file
    LabelMe_file = os.path.join(LabelMe_annotation_dir, 'g16_t1_up', file_+'.xml')
    print LabelMe_file
    #clear existing annotations
    tree = ET.parse(LabelMe_file)
    root = tree.getroot()
    for obj in root.findall('object'):
        root.remove(obj)
    #for each set of coordinates, create object instance
    for index, box in enumerate(filtered_boxes):
        object_ = ET.Element('object')
        root.append(object_)
        name_ = ET.SubElement(object_, 'name')
        name_.text = 'rbc'
        deleted_ = ET.SubElement(object_, 'deleted')
        deleted_.text = "0"
        verified_ = ET.SubElement(object_, 'verified')
        verified_.text = "0"
        occluded_ = ET.SubElement(object_, 'occluded')
        occluded_.text = 'no'
        attributes_ = ET.SubElement(object_, 'attributes')
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
            x_.text = str(box[(i + i%2)%4])
            y_ = ET.SubElement(pt_, 'y')
            y_.text = str(box[i + (i+1)%2])

    tree.write(LabelMe_file)
    os.chmod(LabelMe_file, 0o777)
