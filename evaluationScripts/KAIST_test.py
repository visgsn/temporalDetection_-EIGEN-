'''
    This is the main script for testing the trained KAIST net.
    
    Usage: python KAIST_test.py <GPU-ID for testing>
    IMPORTANT: Also adapt kaist_path path AND kaist_subsets in ./lib/datasets/factory.py
    
    Written by Steffen Guentert
'''

import matlab.engine
import _init_paths
from fast_rcnn.test import single_scale_test_net, multi_scale_test_net_320, multi_scale_test_net_512
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import caffe
import os
import sys
import logging

if __name__ == '__main__':
    ##### CONFIGURATION ################################################################################################
    # Change "atWORK" to switch between HOME and WORK directories (False: HOME - True: WORK)
    atWork  = True  # Change this in "factory.py", too!

    doMatlabEval    = True  # If True: Use additional MATLAB test evaluation (on "test-all")
    redoInference   = False  # If True: Re-execute inference although detecions file already exists (Time consuming)

    subsetName      = "train-all-T"  # Adapt --> "factory.py" for all available subsets!
    job_name        = "refinedet_vgg16_320x320"  # DEFAULT: "refinedet_vgg16_320x320"
    experimentName  = "singleScale"  # Name of evalOutput subfolder (experiment name)
    single_scale    = True  # True: single scale test;  False: multi scale test
    compete_mode    = True  # Specifies evaluation to use UUID (salt) and delete VOC dets afterwards, if False.
    visualizeDets   = False  # Show Detections while testing?

    useGPU          = True  # Default: True

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
    cfg.ROOT_DIR = train_test_outPath  # Defines output directory for evaluation results
    cfg.EXP_DIR = experimentName
    imdb = get_imdb(test_set)  # In this case, imdb is an instance of class "kaist"!
    imdb.competition_mode(compete_mode)
    imdb.set_matlab_eval(evalWithMatlab=doMatlabEval)

    # Set CPU- or GPU-mode
    if useGPU:
        GPU_ID = int(sys.argv[1])  # Adapted to use with script "StartIfGPUFree.py". GPU to use for execution
        caffe.set_mode_gpu()
        caffe.set_device(GPU_ID)
    else:
        caffe.set_mode_cpu()

    if '320' in train_test_outPath:
        input_size = 320
    else:
        input_size = 512

    prototxt = os.path.join(train_test_outPath, 'deploy.prototxt')
    models = sorted(os.listdir(train_test_outPath), key=str.lower)

    mAP = {}    # mean Average Precision
    mPrec = {}  # mean Precision
    mRec = {}   # mean Recall
    miss = {}   # Misses from Matlab
    roc = {}    # Receiver Operating Characteristics from MATLAB (=[score fp tp])
    gt = {}     # Ground Truths from MATLAB
    dt = {}     # Detections from MATLAB
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
            single_scale_test_net(net, imdb, targe_size=input_size, vis=visualizeDets, redoInference=redoInference)
        else:
            if input_size == 320:
                multi_scale_test_net_320(net, imdb, vis=visualizeDets, redoInference=redoInference)
            else:
                multi_scale_test_net_512(net, imdb, vis=visualizeDets, redoInference=redoInference)
        mAP[iter]   = cfg.mAP
        mPrec[iter] = cfg.prec
        mRec[iter]  = cfg.rec
        # Also available for further evaluation (From MATLAB, not used yet):
        #miss[iter]  = cfg.miss
        #roc[iter]   = cfg.roc
        #gt[iter]    = cfg.gt
        #dt[iter]    = cfg.dt

    # Just to check data structure:
    #print("Miss  : " + str( miss.get(miss.keys()[0]) ))
    #print("ROCs  : " + str( roc.get(roc.keys()[0]) ))
    #print("GTs   : " + str( gt.get(gt.keys()[0]) ))
    #print("DTs   : " + str( dt.get(dt.keys()[0]) ))

    keys = mAP.keys()
    keys.sort()
    templine = []
    mAP_outFile = os.path.join(get_output_dir(imdb, forKaist=True), 'mAP_mPrec_mRec.txt')
    print("########################################################################")
    print("########################################################################")
    for key in keys:
        value_mAP   = mAP.get(key)
        value_mPrec = mPrec.get(key)
        value_mRec  = mRec.get(key)
        print("%d\t%.4f\t%.4f\t%.4f" % (key, value_mAP, value_mPrec, value_mRec))
        templine.append("%d\t%.4f\t%.4f\t%.4f\n" % (key, value_mAP, value_mPrec, value_mRec))
    with open(mAP_outFile, 'w+') as f:
        print("\n")
        logging.info("Results can be found under: " + str(train_test_outPath))
        f.writelines(templine)
    print("########################################################################")
    print("########################################################################")