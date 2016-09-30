import pickle
import os
from sys import argv
'''
display results from pkl files
'''
DET = argv[1]
test_name = argv[2]
iterations = argv[3]
THRESH = 0.5
classes = ['__background__', 'rbc', 'tro', 'sch', 'ring', 'gam', 'leu']
base_dir = '/home/ubuntu/py-faster-rcnn/output/faster_rcnn_end2end/'+test_name
base_dir += 'vgg_cnn_m_1024_faster_rcnn_iter_'+iterations
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
