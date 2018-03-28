'''
    This script creates trainval.txt and test.txt, which contain the links between images and their
    corresponding annotations.
    
    To use this script, first run "convertAnnotations.py" to convert the necessary Annotations
    from the kaist-data sub-folder into the KAIST folder (target format: .xml).

    IMPORTANT: Check paths below before execution!
               "outputFolder" has to be the location of the previously generated "Annotations" folder!!!
'''

from _usefulFunctions import *
import os
import logging
#import random


##### CONFIGURATION #######################################################
### *** HOME ***
kaistFolder     = '/home/gueste/data/KAIST/data-kaist'
outputFolder    = '/home/gueste/data/KAIST'
### *** WORK ***
#kaistFolder     = '/net4/merkur/storage/deeplearning/users/gueste/data/KAIST/data-kaist'
#outputFolder    = '/net4/merkur/storage/deeplearning/users/gueste/data/KAIST'

trainImgSub     = 'train-all-T'     # Desired training subset of kaistFolder (Contains .png IMAGES!)
testImgSub      = 'test-all'        # Desired test subset of kaistFolder (Contains .png IMAGES!)

useThermal      = True              # If False, 'RGB_'-images will be extracted.
regexThermal    = 'T_.*.png$'
regexRgb        = 'RGB_.*.png$'

logging.basicConfig(format='%(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
###########################################################################



trainFolder = os.path.join(kaistFolder, trainImgSub, "images")
testFolder = os.path.join(kaistFolder, testImgSub, "images")


# Create trainval.txt
trainValFileName = os.path.join(outputFolder, "ImageSets/Main/trainval.txt")
trainValFilePath = os.path.split(trainValFileName)[0]
logging.info("Creating trainVal file: " + str(trainValFileName))

if not os.path.isdir(trainValFilePath):
    logging.info("Creating output directory '" + str(trainValFilePath) + "' because it doesn't exist.")
    os.makedirs(trainValFilePath)

with open(trainValFileName, 'w+') as trainValFile:
    if useThermal:
        logging.info("Using THERMAL images from: " + str(trainFolder))
        files = dirRecursive(trainFolder, regexThermal) # "T_" --> Thermal imag.
    else:
        logging.info("Using RGB images from: " + str(trainFolder))
        files = dirRecursive(trainFolder, regexRgb)     # "RGB_" --> Colored imag.

    for filename in files:    # List of all files in trainFolder
        _,filename,_ = fileParts(filename)
        if os.path.splitext(filename)[1] != '.db':
            trainValFile.write(trainFolder + '/' + filename + '.png ')
            trainValFile.write(outputFolder + '/Annotations/' + filename + '.xml')
            trainValFile.write("\n")


# Create test.txt
testFileName = os.path.join(outputFolder, "ImageSets/Main/test.txt")
testFilePath = os.path.split(trainValFileName)[0]
logging.info("Creating test file: " + str(testFileName))

if not os.path.isdir(testFilePath):
    logging.info("Creating output directory '" + str(testFilePath) + "' because it doesn't exist.")
    os.makedirs(testFilePath)

with open(testFileName, 'w+') as testFile:
    if useThermal:
        logging.info("Using THERMAL images from: " + str(testFolder))
        files = dirRecursive(testFolder, regexThermal)  # "T_" --> Thermal imag.
    else:
        logging.info("Using THERMAL images from: " + str(testFolder))
        files = dirRecursive(testFolder, regexRgb)      # "RGB_" --> Colored imag.

    for filename in files:     # List of all files in testFolder
        _,filename,_ = fileParts(filename)
        if os.path.splitext(filename)[1] != '.db':
            testFile.write(testFolder + '/' + filename + '.png ')
            testFile.write(outputFolder + '/Annotations/' + filename + '.xml')
            testFile.write("\n")

print "\n\n"
logging.info("DONE")
