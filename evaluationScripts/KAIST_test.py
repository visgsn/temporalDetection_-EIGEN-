import _init_paths
from fast_rcnn.test import single_scale_test_net, multi_scale_test_net_320, multi_scale_test_net_512
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import os
import sys

if __name__ == '__main__':
    ##### CONFIGURATION ################################################################################################
    ### *** HOME ***
    GPU_ID = 0
    path_prefix = "{}/train_test_data".format(os.environ['HOME'])
    ### *** WORK ***
    #GPU_ID = int(sys.argv[1])   #Adapted to use with script "StartIfGPUFree.py". GPU to use for execution
    #path_prefix = "/net4/merkur/storage/deeplearning/users/gueste/TRAINING_test"

    test_set = 'voc_2007_test' # 'voc_2007_test' or 'voc_2012_test' or 'coco_2014_minival' or 'coco_2015_test-dev'
    single_scale = True # True: single scale test;  False: multi scale test
    ####################################################################################################################

    kaist_path = '{}/models/VGGNet/VOC0712/refinedet_vgg16_320x320/'.format(path_prefix)

    cfg.single_scale_test = single_scale

    if '320' in kaist_path:
        input_size = 320
    else:
        input_size = 512

    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)

    imdb = get_imdb(test_set)
    imdb.competition_mode(False)

    #prototxt = kaist_path + 'deploy.prototxt'
    prototxt = os.path.join(kaist_path, 'deploy.prototxt')
    models = os.listdir(kaist_path)

    mAP = {}    # mean Average Precision
    for model in models:
        if model.find('caffemodel') == -1:
            continue
        caffemodel = kaist_path + model
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(model))[0]
        cfg.net_name = net.name
        try:
            iter = int(net.name.split('_')[-1])
        except:
            iter = 000000
        if single_scale is True:
            single_scale_test_net(net, imdb, targe_size=input_size)
        else:
            if input_size == 320:
                multi_scale_test_net_320(net, imdb)
            else:
                multi_scale_test_net_512(net, imdb)
        mAP[iter] = cfg.mAP

    keys = mAP.keys()
    keys.sort()
    templine = []
    print("########################################################################")
    print("########################################################################")
    for key in keys:
        value = mAP.get(key)
        print("%d\t%.4f"%(key, value))
        templine.append("%d\t%.4f\n"%(key, value))
    with open(kaist_path + 'mAP.txt', 'w+') as f:
        f.writelines(templine)
    print("########################################################################")
    print("########################################################################")