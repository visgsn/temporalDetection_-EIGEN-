'''
    This script is used to copy and rename the images and annotations from the KAIST dataset, which have previously
    been extracted with Pjotr's Matlab Toolbox. (By default only training data from set00 to set05)
    Names are chosen according to the scheme of "train_all20".
    Example for images:
        T_tmp_set00_V000_100019_<UUID>.png         (T_ stands for "thermal", RGB_ for color images)
    Example for annotations:
        set00_V000_100019.txt
'''

from _usefulFunctions import *
import os
import sys
import re
import logging

##### Configurations ########################################
### *** HOME ***
annoDir     = "/home/gueste/data/KAIST/data-kaist/annotations-extracted"    # Input annotations
imageDir    = "/home/gueste/data/KAIST/data-kaist/videos-extracted"         # Input images
outputDir   = "/home/gueste/data/KAIST/data-kaist/train-all"
### *** WORK ***
# Insert config for WORK here...

logging.basicConfig(format='%(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
annoRegex       = '.*.txt$'         # Used to find available annotations
imageRegex      = 'frame_.*.png$'   # Used to find available images
extractSetRegex = '.*\/set0[0-5]\/' # Specifies which sets to extract (Default: '.*\/set0[0-5]\/')
splitThermRegex = '.*\/LWIR_V'      # Used to separate RGB- and thermal images
#############################################################



### MAIN
# Construct data paths and create output directory
annoDir_out     = os.path.join(outputDir, "annotations")
imageDir_out    = os.path.join(outputDir, "images")
if not os.path.isdir(annoDir_out):
    logging.info("Creating output directory '" + str(annoDir_out) + "' because it doesn't exist.")
    os.makedirs(annoDir_out)
if not os.path.isdir(imageDir_out):
    logging.info("Creating output directory '" + str(imageDir_out) + "' because it doesn't exist.")
    os.makedirs(imageDir_out)

# Collect available images and annotations
availableAnnos  = dirRecursive(annoDir, annoRegex)
availableImages = dirRecursive(imageDir, imageRegex)
# Sort arrays
availableAnnos     = sorted(availableAnnos, key=str.lower)
availableImages    = sorted(availableImages, key=str.lower)
logging.info("Found " + str(len(availableAnnos)) + " available annotations in total, e.g.:\n" \
             + str(availableAnnos[0:5]))
logging.info("---------------------------------------------------------------------------------")
logging.info("Found " + str(len(availableImages)) + " available images in total, e.g.:\n" \
             + str(availableImages[0:5]) + "\n")

# Pick desired sets from database (Extract only training sets)
setPattern  = re.compile(extractSetRegex)
temp_var    = []
for singleFile in availableAnnos:
    if setPattern.match(singleFile):
        temp_var.append(singleFile)
availableAnnos = temp_var[:]
temp_var    = []
for singleFile in availableImages:
    if setPattern.match(singleFile):
        temp_var.append(singleFile)
availableImages = temp_var[:]
logging.info("Found " + str(len(availableAnnos)) + " annotations after choosing sets, e.g.:\n" \
             + str(availableAnnos[0:5]))
logging.info("---------------------------------------------------------------------------------")
logging.info("Found " + str(len(availableImages)) + " images after choosing sets, e.g.:\n" \
             + str(availableImages[0:5]) + "\n")

# Separate RGB- and thermal images
thermPattern      = re.compile(splitThermRegex)
thermalImages   = []
rgbImages       = []
for singleFile in availableImages:
    if thermPattern.match(singleFile):
        thermalImages.append(singleFile)
    else:
        rgbImages.append(singleFile)
# Check if images are equally split
if len(thermalImages) != len(availableAnnos) or len(rgbImages) != len(availableAnnos):
    logging.warning("WARNING: #images doesn't match #annotations!")
elif len(thermalImages) != len(rgbImages):
    logging.warning("WARNING: Images not equally split!")
logging.info("Found " + str(len(availableAnnos)) + " annotations after splitting images, e.g.:\n" \
             + str(availableAnnos[0:5]))
logging.info("---------------------------------------------------------------------------------")
logging.info("Found " + str(len(thermalImages)) + "/" + str(len(rgbImages)) + \
             " (THERMAL/RGB) images after splitting images, e.g.:\n" \
             "Thermal: " + str(thermalImages[0:5]) + "\n" \
             "RGB    : " + str(rgbImages[0:5]) + "\n")



# Split images and annotations
