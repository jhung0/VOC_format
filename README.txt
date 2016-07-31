Workflow to prepare data (annotations, images) to be in right format

From the LabelMe version
Collection
    Annotations
        users
            jane24
                collectionName
                    *.xml
    Images
        users
            jane24
                collectionName
                    *.jpg

1. Replace .jpg files with .tif files
2. Only keep .tif files that have an associated Annotation file
in the Images -> users -> jane24 -> collectionName directory run
python remove_images.py [directory where replacement .tif images are] [directory where images should be]
python remove_images.py /opt/NF54gamplusring_20151223/ /home/jyhung/Downloads/collection/Images/users/jane24/54gamplusring_20151223_outlines/ 

3. Convert from LabelMe format to VOC format
4. Cut each image into subimages, creating new annotation files with adjusted boxes
python VOC_format.py [output directory] [image files in Images]
e.g. python VOC_format.py ../try1/data/ ../Downloads/collection/Images/users/jane24/54gamplusring_20151223_outlines/*
*change output_dir, num_subimages, small_size, code under #choose whether file is part of training or testing set

e.g. python VOC_format3.py ../try1/data/ /var/www/html/LabelMeAnnotationTool/Images/g16_t1_up g8_t1_up g7_t1_up

VOC_format
    data
        Annotations
            *.txt
        ImageSets
            train.txt
            test.txt
        Images
            *.tif
5. Move data to github repo

6. Change faster rcnn code:
lib/datasets/try1.py

models/VGG_CNN_M_1024/train.prototxt or
models/VGG_CNN_M_1024/stage1_fast_rcnn_train.pt, stage1_rpn_train.pt, stage2_fast_rcnn_train.pt, stage2_rpn_train.pt, and faster_rcnn_test.pt

remove
data/cache/train_gt_roidb.pkl
