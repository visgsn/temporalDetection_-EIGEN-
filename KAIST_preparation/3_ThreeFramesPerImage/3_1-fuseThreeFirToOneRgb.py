'''
    This script converts all annotations from the KAIST dataset to a compatible format for VOC
    and exports them as .xml files into the output folder (renamed according to image name).

    Afterwards you can run "generateVocLists.py"

    Check paths below before execution!
'''

import os
import sys
import logging
import shutil
import numpy as np
import cv2
import re
sys.path.append( os.path.abspath(os.path.join(os.path.curdir, "..")) )  # Added to enable import from parent directory
from _usefulFunctions import *



##### Configurations ###################################################################################################
atWORK          = False  # Choose which config to use: HOME (False) - WORK (True)

dataToExtract   = ['train-all-T', 'test-all-T']  # Expects: [<Train_Set>, <Test_Set>] (OutputSubdir <-- Train_Set name)
imageStepSize   = 5  # Distance between images in channel (R=t, G=t-1*iSS, B=t-2*iSS)

kaistDirHOME    = '/home/gueste/data/KAIST/data-kaist'
kaistDirWORK    = '/net4/merkur/storage/deeplearning/users/gueste/data/KAIST/data-kaist'

mainTestSet     = 'test-all'  # Main test set for comparison (Do not change this for KAIST dataset!)
useThermal      = True  # If False, 'RGB_'-images will be extracted.

set_V_Pattern   = '(set[0-9]+_V[0-9]+)_'  # Used for comparison of predecessor image names with original image
logging.basicConfig(format='%(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
########################################################################################################################

dataToExtract   = ['test-all-T', 'train-all-T']                                                                         # Wieder entfernen!!!

kaistDir = kaistDirWORK[:] if atWORK else kaistDirHOME[:]
assert os.path.exists(kaistDir), \
    "Path {} does not exist! --> atWORK = ?".format(kaistDir)

### Scan official test data for available annotation and names (e.g. "set06_V003_I04219.txt")
mainTestDir = os.path.join(kaistDir, mainTestSet)
mainAnnoDir = os.path.join(mainTestDir, 'annotations')
mainImgDir  = os.path.join(mainTestDir, 'images')
assert os.path.exists(mainTestDir), \
    "KAIST main test set folder does not exist under '{}'!".format(mainTestDir)
mainAnnoFiles   = dirRecursive(mainAnnoDir, '.*.txt$')
# Extract annotation / image names
for i, singleMainAnno in enumerate(mainAnnoFiles):
    _,mainName,_ = fileParts(singleMainAnno)
    mainAnnoFiles[i] = mainName

### Copy necessary files from input datasets (see: dataToExtract) and execute image fusion
for folder in dataToExtract:
    # Construct data output paths and create output directories
    outputDir = os.path.join(kaistDir, '3_{}_D{}'.format(folder, imageStepSize))                                        # Output path format!
    annoDir_out = os.path.join(outputDir, "annotations")
    imageDir_out = os.path.join(outputDir, "images")
    if not os.path.isdir(outputDir):
            logging.info("Creating output directory '" + str(annoDir_out) + "' because it doesn't exist.")
            os.makedirs(annoDir_out)
            logging.info("Creating output directory '" + str(imageDir_out) + "' because it doesn't exist.")
            os.makedirs(imageDir_out)
    else:
        logging.error("ERROR: Output set path '{}' already existing!".format(outputDir))
        sys.exit()

    # Search for existing annotation and image files
    inputDir    = os.path.join(kaistDir, folder)
    annoDir_in  = os.path.join(inputDir, 'annotations')
    imageDir_in = os.path.join(inputDir, 'images')

    annoFiles_in = dirRecursive(annoDir_in, '.*.txt$')
    if useThermal:
        imgFiles_in = dirRecursive(imageDir_in, 'T_.*.png$')   # "T_" --> Thermal imag.
    else:
        imgFiles_in = dirRecursive(imageDir_in, 'RGB_.*.png$') # "RGB_" --> Colored imag.

    # Check for equal number of images and annotations
    assert len(annoFiles_in) == len(imgFiles_in), \
        "Number of annotations not equal to number of images!"

    # Test data --> Copy only files that also exist in official the mainTestSet, afterwards fuse them as desired
    annoFiles_out   = []
    imgFiles_out    = []
    imgIndex_out    = []  # Denotes the position of an image in the original imgFiles_in array
    if 'test' in folder:
        logging.info("Copying TEST data from '{}' to '{}'".format(inputDir, outputDir))
        foundFlag   = False
        nextIter    = 0  # Used to speed up search
        for name in mainAnnoFiles:
            for i in range(nextIter, len(annoFiles_in)):
                if name in os.path.split(annoFiles_in[i])[1] and name in os.path.split(imgFiles_in[i])[1]:
                    annoFiles_out.append(annoFiles_in[i])
                    imgFiles_out.append(imgFiles_in[i])
                    imgIndex_out.append(i)
                    foundFlag = True  # Set flag -> corresponding annotation / image pair found
                    nextIter = i + 1
                    break
            # Check if pair found
            assert foundFlag == True, "No corresponding anno / img pait found for '{}'".format(name)
            foundFlag = False  # Reset flag

    # Train data --> Copy all files and fuse them as desired
    else:
        logging.info("Copying TRAIN data from '{}' to '{}'".format(inputDir, outputDir))
        annoFiles_out   = annoFiles_in[:]
        imgFiles_out    = imgFiles_in[:]
        imgIndex_out    = range(len(imgFiles_in))

    ### Determine predecessors of images which have to be combined (fused) to one single image in RGB format
    for i, singleImg in enumerate(imgFiles_out):
        tmpImgFiles = [singleImg, '', '']  # Stores original image path and its 2 predecessors
        for predNr in range(1,3):
            predIndex = imgIndex_out[i] - (predNr * imageStepSize)
            if predIndex >= 0:
                # Determine filenames for comparison of setXX_VXXX
                _,origName,_ = fileParts(singleImg)
                _,predName,_ = fileParts(imgFiles_in[predIndex])
                origSetV = re.search(set_V_Pattern, origName).group(1)
                predSetV = re.search(set_V_Pattern, predName).group(1)
                if origSetV == predSetV:
                    # Set calculated predecessor (image content temporally connected) and continue with next iteration
                    tmpImgFiles[predNr] = imgFiles_in[predIndex]
                    continue
            # Set same predecessor as before (image content NOT temporally connected or index out of range)
            tmpImgFiles[predNr] = tmpImgFiles[predNr-1]

        # Assign predecessors behind original image
        imgFiles_out[i] = tmpImgFiles[:]

    ### Fuse and save images
    for i in range(0, len(imgFiles_out)):
        # Load original image (BGR-Format: BGR --> 012)
        imgOrig = cv2.imread(imgFiles_out[i][0])
        # Replace channels G (1) and B (0) of original with predecessors R (2) channel
        for predNr in range(1,3):
            imgPred = cv2.imread(imgFiles_out[i][predNr])
            imgOrig[:,:,2-predNr] = imgPred[:,:,2]
        # Save resulting image in target folder
        outputFileName = os.path.join( imageDir_out, os.path.split(imgFiles_out[i][0])[1] )
        cv2.imwrite(outputFileName, imgOrig)

    ### Copy annotations
    # Annotations
    for singleAnno in annoFiles_out:
        shutil.copy(singleAnno, annoDir_out)
