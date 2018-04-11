'''
    This script creates trainval.txt, test.txt and test_name_size.txt, which contain the links between images and their
    corresponding annotations, as well as image size information.
    
    To use this script, first run "convertAnnotations.py" to convert the necessary Annotations
    from the kaist-data sub-folder into the KAIST folder (target format: .xml).

    IMPORTANT: Check paths below before execution!
               "kaistDataFolder" has to point to the location of the "data-kaist" folder inside KAIST!!!
'''

from _usefulFunctions import *
import os
import logging
from PIL import Image
from random import shuffle

##### CONFIGURATION ####################################################################################################
atWORK          = True  # Choose which config to use: HOME (False) - WORK (True)

dataKaistHOME   = '/home/gueste/data/KAIST/data-kaist'
dataKaistWORK   = '/net4/merkur/storage/deeplearning/users/gueste/data/KAIST/data-kaist'

trainImgSub     = 'train-all-T'     # Desired training subset of dataKaist (Contains .png IMAGES!)
testImgSub      = 'test-all'        # Desired test subset of dataKaist (Contains .png IMAGES!)

useThermal      = True              # If False, 'RGB_'-images will be extracted.
shuffleFiles    = True              # Randomize order of train images
regexThermal    = 'T_.*.png$'
regexRgb        = 'RGB_.*.png$'

logging.basicConfig(format='%(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
########################################################################################################################



dataKaist = dataKaistWORK[:] if atWORK else dataKaistHOME
assert os.path.exists(dataKaist), \
    "Path {} does not exist! --> atWORK = ?".format(dataKaist)

kaistRoot           = os.path.split(dataKaist)[0]
outputFolder        = os.path.join(kaistRoot, trainImgSub)
outputFolder_rel    = os.path.relpath(outputFolder, kaistRoot)
trainFolder         = os.path.join(dataKaist, trainImgSub, "images")
trainFolder_rel     = os.path.relpath(trainFolder, kaistRoot)
testFolder          = os.path.join(dataKaist, testImgSub, "images")
testFolder_rel      = os.path.relpath(testFolder, kaistRoot)

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

    # Shuffle train images?
    logging.info("Randomize train image order: " + ("YES" if shuffleFiles else "NO") + "\n")
    if shuffleFiles: shuffle(files)

    for filename in files:    # List of all files in trainFolder
        if os.path.splitext(filename)[1] != '.db':
            _,filename,_ = fileParts(filename)
            trainValFile.write(trainFolder_rel + '/' + filename + '.png ')
            trainValFile.write(outputFolder_rel + '/Annotations/' + filename + '.xml')
            trainValFile.write("\n")


# Create test.txt AND test_name_size.txt
testFileName = os.path.join(outputFolder, "ImageSets/Main/test.txt")
testFilePath = os.path.split(trainValFileName)[0]
nameSizeFileName = os.path.join(testFilePath, "test_name_size.txt")
logging.info("Creating test file:  " + str(testFileName) + " and\n" + \
             "test_name_size file: " + str(nameSizeFileName))

if not os.path.isdir(testFilePath):
    logging.info("Creating output directory '" + str(testFilePath) + "' because it doesn't exist.")
    os.makedirs(testFilePath)

with open(testFileName, 'w+') as testFile, open(nameSizeFileName, 'w+') as nameSizeFile:
    if useThermal:
        logging.info("Using THERMAL images from: " + str(testFolder))
        files = dirRecursive(testFolder, regexThermal)  # "T_" --> Thermal imag.
    else:
        logging.info("Using THERMAL images from: " + str(testFolder))
        files = dirRecursive(testFolder, regexRgb)      # "RGB_" --> Colored imag.

    for filename in files:     # List of all files in testFolder
        if os.path.splitext(filename)[1] != '.db':
            # Get image size
            im = Image.open(filename)
            width, height = im.size
            # Separate file name
            _,filename,_ = fileParts(filename)

            # Write test file
            testFile.write(testFolder_rel + '/' + filename + '.png ')
            testFile.write(outputFolder_rel + '/Annotations/' + filename + '.xml')
            testFile.write("\n")
            # Write test_name_size file
            nameSizeFile.write(filename + " " + str(height) + " " + str(width))
            nameSizeFile.write("\n")


print "\n"
logging.info("DONE")
