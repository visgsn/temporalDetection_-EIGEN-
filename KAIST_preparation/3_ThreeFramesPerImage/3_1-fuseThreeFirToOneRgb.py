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
sys.path.append( os.path.abspath(os.path.join(os.path.curdir, "..")) )  # Added to enable import from parent directory
from _usefulFunctions import *



##### Configurations ###################################################################################################
atWORK          = False  # Choose which config to use: HOME (False) - WORK (True)

dataToExtract   = ['train-all-T', 'test-all-T']  # Expects: [<Train_Set>, <Test_Set>] (OutputSubdir <-- Train_Set name)
imageStepSize   = 5  # Distance between images in channel (R=t, G=t-1*iSS, B=t-2*iSS)
useThermal      = True  # If False, 'RGB_'-images will be extracted.

kaistDirHOME    = '/home/gueste/data/KAIST/data-kaist'
kaistDirWORK    = '/net4/merkur/storage/deeplearning/users/gueste/data/KAIST/data-kaist'

mainTestSet     = 'test-all'  # Main test set for comparison (Do not change this for KAIST dataset!)

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

    annoFiles = dirRecursive(annoDir_in, '.*.txt$')
    if useThermal:
        imgFiles = dirRecursive(imageDir_in, 'T_.*.png$')   # "T_" --> Thermal imag.
    else:
        imgFiles = dirRecursive(imageDir_in, 'RGB_.*.png$') # "RGB_" --> Colored imag.

    # Check for equal number of images and annotations
    assert len(annoFiles) == len(imgFiles), \
        "Number of annotations not equal to number of images!"

    # Test data --> Copy only files that also exist in official the mainTestSet, afterwards fuse them as desired
    if 'test' in folder:
        logging.info("Copying TEST data from '{}' to '{}'".format(inputDir, outputDir))
        tmpAnno     = []
        tmpImg      = []
        foundFlag   = False
        lastIter    = 0  # Used to speed up search
        for name in mainAnnoFiles:
            for i in range(lastIter, len(annoFiles)):
                if name in os.path.split(annoFiles[i])[1]:
                    tmpAnno.append(annoFiles[i])
                    tmpImg.append(imgFiles[i])
                    foundFlag = True  # Set flag -> corresponding annotation / image pair found
                    lastIter = i
                    break
            # Check if pair found
            assert foundFlag == True, "No corresponding anno / img pait found for '{}'".format(name)
            foundFlag = False  # Reset flag
        annoFiles   = tmpAnno[:]
        imgFiles    = tmpImg[:]

    # Train data --> Copy all files and fuse them as desired
    else:
        logging.info("Copying TRAIN data from '{}' to '{}'".format(inputDir, outputDir))

    ### Determine images which have to be combined (fused) to one single image in RGB format
    

    ### Copy and fused images
    # Annotations
    for singleAnno in annoFiles:
        shutil.copy(singleAnno, annoDir_out)
    # Images
    for singleImg in imgFiles:
        shutil.copy(singleImg, imageDir_out)
