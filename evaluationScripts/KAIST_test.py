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

    # Adapt --> "factory.py" <-- for all available subsets!
    #--------------------SUBSET----------------------Job----------------------------------------------------------------
    subset_and_job  = [#["train-all-T",              "refinedet_50home_320x320"],
                       #["train-all-T",              "refdet_i40k_Adam_DROPOUT_512x512"],
                       #["train-all-T",              "refdet_i100k_Adam_512x512"],
                       #["train-all-T",              "refdet_i100k_Adam_DROPOUT_512x512"],
                       #["train-all-T",              "refdet_i200k_lr001_320x320"],
                       #["train-all-T",              "refdet_i200k_lr001_512x512"],
                       #["train-all-T",              "refdet_i200k_lr001_DROPOUT_320x320"],
                       #["train-all-T",              "refdet_i200k_lr001_DROPOUT_512x512"],
                       #["train-all-T",              "refdet_i200k_lr001_DROPOUT_NEW_512x512"],
                       #["train-all-T",              "refdet_i10k_DROPOUT_test_320x320"],
                       #["train-all-T",              "refinedet_it50184_320x320"],
                       #["train-all-T",              "Tr10_i40k_Adam_DROP_lr0001STEP_512x512"],
                       #["train-all-T",              "Tr11_HOME_i10k_Adam_512x512"],
                       #["train-all-T",              "Tr12_i20k_Adam_512x512"],
                       #["train-all-T",              "Tr13_NEW_i20k_lr001_bs180_512x512"],
                       #
                       #
                       #["2_train-all-T_D4",         "2_Tr1-2_OF_D4_320x320"],          # MULTI_SCALE TEST!
                       #["2_train-all-T_D4",         "2_Tr2-2_OF_D4_320x320"],
                       #["2_train-all-T_D4",         "2_Tr3_OF_D4_512x512"],
                       #["2_train-all-T_D4",         "2_Tr4_OF_D4_512x512"],            # MULTI_SCALE TEST!
                       #["2_train-all-T_D4",         "2_Tr5_OF_D4_320x320"],
                       #["2_train-all-T_D4",         "2_Tr6_OF_D4_320x320"],
                       #["2_train-all-T_D4",         "2_Tr7_OF_D4_320x320"],
                       #["2_train-all-T_D4",         "2_Tr8_OF_D4_320x320"],
                       #["2_train-all-T_D4",         "2_Tr9_OF_D4_320x320"],
                       #["2_train-all-T_D4",         "2_Tr10_OF_D4_320x320"],
                       #["2_train-all-T_D1",         "2_Tr15-1_OF_D1_320x320"],
                       ["2_train-all-T_D1",         "2_Tr15-2_OF_D1_320x320"],     #
                       #["2_train-all-T_D7",         "2_Tr16-1_OF_D7_320x320"],
                       ["2_train-all-T_D7",         "2_Tr16-2_OF_D7_320x320"],     #
                       #["2_train-all-T_D10",        "2_Tr17-1_OF_D10_320x320"],
                       ["2_train-all-T_D10",        "2_Tr17-2_OF_D10_320x320"],    #
                       ["2_train-all-T_D4",         "2_Tr18-2_OF_D4_320x320"],     #
                       #
                       #
                       #["3_train-all-T_D4",         "3_Tr1_3FpI_i60k_320x320"],
                       #["3_train-all-T_D4",         "3_Tr2_3FpI_D4_512x512"],
                       #["3_train-all-T_D4",         "3_Tr3_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr4_3FpI_D4_512x512"],
                       #["3_train-all-T_D4",         "3_Tr5_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr6_3FpI_D4_512x512"],
                       #["3_train-all-T_D4",         "3_Tr7_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr7_3FpI_D4_TEST1_320x320"],
                       #["3_train-all-T_D4",         "3_Tr7_3FpI_D4_TEST2_320x320"],
                       #["3_train-all-T_D4",         "3_Tr8_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr9-2_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr10-2_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr11_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr12-2_3FpI_D4_320x320"],      # MULTI_SCALE TEST!
                       #["3_train-all-T_D4",         "3_Tr13-2_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr14-1_3FpI_D4_320x320"],      # MULTI_SCALE TEST!
                       #["3_train-all-T_D4",         "3_Tr15_3FpI_D4_512x512"],
                       #["3_train-all-T_D4",         "3_Tr16_3FpI_D4_512x512"],
                       #["3_train-all-T_D4",         "3_Tr17_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr18_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr19_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr20_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr21_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr22_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr23_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr24_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr25_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr26_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr27_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr28_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr29_3FpI_D4_320x320"],
                       #["3_train-all-T_D4",         "3_Tr30_3FpI_D4_512x512"],
                       #["3_train-all-T_D4",         "3_Tr31_3FpI_D4_512x512"],
                       #["3_train-all-T_D1",         "3_Tr35-1_3FpI_D1_320x320"],
                       ["3_train-all-T_D1",         "3_Tr35-2_3FpI_D1_320x320"],    #
                       #["3_train-all-T_D7",         "3_Tr36-1_3FpI_D7_320x320"],
                       ["3_train-all-T_D7",         "3_Tr36-2_3FpI_D7_320x320"],    #
                       #["3_train-all-T_D10",        "3_Tr37-1_3FpI_D10_320x320"],
                       ["3_train-all-T_D10",        "3_Tr37-2_3FpI_D10_320x320"],   #
                       ]  # DEFAULT: ["refinedet_vgg16_320x320"]

    single_scale    = True  # True: single scale test;  False: multi scale test
    compete_mode    = True  # Specifies evaluation to use UUID (salt) and delete VOC dets afterwards, if False.
    visualizeDets   = False  # Show Detections while testing?
    # Optional postfix for evalOutput subfolder (experiment name) -> If set: <experimentName>_<expNamePostfix>
    expNamePostfix  = ""  # DEFAULT: ""

    useGPU          = True  # Default: True

    path_prefix_HOME = "{}/train_test_data".format(os.environ['HOME'])
    path_prefix_WORK = "/net4/merkur/storage/deeplearning/users/gueste/TRAINING_test"

    logging.basicConfig(format='%(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    ####################################################################################################################


    path_prefix = path_prefix_WORK if atWork else path_prefix_HOME
    assert os.path.exists(path_prefix), \
        "Path {} does not exist! --> atWORK = ?".format(path_prefix)
    # Automatically set experiment name
    experimentName = "singleScale" if single_scale else "multiScale"
    if expNamePostfix != "":
        experimentName = "{}_{}".format(experimentName, expNamePostfix)  # Add Postfix, if set


    for subsetName, job_name in subset_and_job:
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
        miss = {}   # Log-Average Miss Rate from Matlab
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
            miss[iter]  = cfg.miss
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
        mAP_outFile = os.path.join(get_output_dir(imdb, forKaist=True), 'mAP_mPrec_mRec_laMiss.txt')
        print("########################################################################")
        print("########################################################################")
        print("KEY   \tmAP   \tmPrec \tmRec  \tlaMiss:")
        for key in keys:
            value_mAP   = mAP.get(key)
            value_mPrec = mPrec.get(key)
            value_mRec  = mRec.get(key)
            value_miss  = miss.get(key)
            print("%d\t%.4f\t%.4f\t%.4f\t%.4f" % (key, value_mAP, value_mPrec, value_mRec, value_miss))
            templine.append("%d\t%.4f\t%.4f\t%.4f\t%.4f\n" % (key, value_mAP, value_mPrec, value_mRec, value_miss))
        with open(mAP_outFile, 'w+') as f:
            print("\n")
            logging.info("Results can be found under: " + str(train_test_outPath))
            f.writelines(templine)
        print("########################################################################")
        print("########################################################################")
