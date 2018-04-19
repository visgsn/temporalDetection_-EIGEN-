'''
    This is the main script for testing the trained KAIST net.
    
    Usage: python KAIST_test.py <GPU-ID for testing>
    IMPORTANT: Also adapt kaist_path path AND kaist_subsets in ./lib/datasets/factory.py
    
    Written by Steffen Guentert
'''

import _init_paths
from fast_rcnn.test import single_scale_test_net, multi_scale_test_net_320, multi_scale_test_net_512
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import os
import sys
import logging

if __name__ == '__main__':
    ##### CONFIGURATION ################################################################################################
    # Change "atWORK" to switch between HOME and WORK directories (False: HOME - True: WORK)
    atWork  = True  # Change this in "factory.py", too!

    MatlabEval  = True  # If True: Use additional MATLAB test evaluation

    subsetName      = "train-all-T"  # Adapt --> "factory.py" for all available subsets!
    job_name        = "refinedet_vgg16_320x320"  # DEFAULT: "refinedet_vgg16_320x320"
    single_scale    = True  # True: single scale test;  False: multi scale test
    compete_mode    = True  # Specifies evaluation to use UUID (salt) and delete VOC dets afterwards, if False.
    GPU_ID          = int(sys.argv[1])  # Adapted to use with script "StartIfGPUFree.py". GPU to use for execution

    path_prefix_HOME = "{}/train_test_data".format(os.environ['HOME'])
    path_prefix_WORK = "/net4/merkur/storage/deeplearning/users/gueste/TRAINING_test"

    logging.basicConfig(format='%(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    ####################################################################################################################


    path_prefix = path_prefix_WORK if atWork else path_prefix_HOME
    assert os.path.exists(path_prefix), \
        "Path {} does not exist! --> atWORK = ?".format(path_prefix)


    train_test_outPath = '{}/models/VGGNet/KAIST/{}/{}/'.format(path_prefix, subsetName, job_name)
    test_set = 'kaist_{}_test'.format(subsetName) # Available: 'kaist_Train-all-T_test' or ... (See --> factory.py for all)

    # Set configuration options
    cfg.single_scale_test = single_scale
    cfg.ROOT_DIR    = train_test_outPath  # Defines output directory for evaluation results

    if '320' in train_test_outPath:
        input_size = 320
    else:
        input_size = 512

    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)

    imdb = get_imdb(test_set)  # In this case, imdb is an instance of class "kaist"!
    imdb.competition_mode(compete_mode)
    imdb.set_matlab_eval(evalWithMatlab=MatlabEval)

    prototxt = os.path.join(train_test_outPath, 'deploy.prototxt')
    models = os.listdir(train_test_outPath)

    mAP = {}    # mean Average Precision
    for model in models:
        if model.find('caffemodel') == -1:
            continue
        caffemodel = train_test_outPath + model
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
    with open(train_test_outPath + 'mAP.txt', 'w+') as f:
        f.writelines(templine)
    print("########################################################################")
    print("########################################################################")