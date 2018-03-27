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
thermPattern    = re.compile(splitThermRegex)
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


# Combine new file names
nameAnno    = []
nameThermal = []
nameRgb     = []
for (i,list_entry) in enumerate(availableAnnos):
    # Split path until all necessary variables are filled
    list_entry = os.path.splitext(list_entry)   # (path/setXX/VXXX/filename, .extension)
    extensionAnno = list_entry[1]
    logging.debug("ExtensionAnno: " + str(extensionAnno))

    extensionTherm = os.path.splitext(thermalImages[i])[1]
    logging.debug("ExtensionTherm: " + str(extensionTherm))

    extensionRgb = os.path.splitext(rgbImages[i])[1]
    logging.debug("ExtensionRgb: " + str(extensionRgb))

    list_entry = os.path.split(list_entry[0])   # (path/setXX/VXXX, filename)
    temp_var = re.search('I([0-9]+)', list_entry[1])
    frameNr = temp_var.group(1)
    logging.debug("Frame #: " + str(frameNr))   # Without first number!!! (I)

    list_entry = os.path.split(list_entry[0])   # (path/setXX, VXXX)
    temp_var = re.search('(V[0-9]+)', list_entry[1])
    subset = temp_var.group(0)
    logging.debug("Subset: " + str(subset))

    list_entry = os.path.split(list_entry[0])   # (path, setXX)
    temp_var = re.search('(set[0-9]+)', list_entry[1])
    set = temp_var.group(0)
    logging.debug("Set: " + str(set))

    # Wrap single parts together to one big filename for each filetype
    nameAnno    = "{}_{}_I{}{}".format(set, subset, frameNr, extensionAnno)
    nameThermal = "T_tmp_{}_{}_I{}{}".format(set, subset, frameNr, extensionTherm)
    nameRgb     = "RGB_tmp_{}_{}_I{}{}".format(set, subset, frameNr, extensionRgb)
    logging.debug("nameAnno     : " + str(nameAnno))
    logging.debug("nameThermal  : " + str(nameThermal))
    logging.debug("nameRgb      : " + str(nameRgb))




    # path = splitFP[0]
    # name = os.path.splitext(splitFP[1])[0]
    #
    # print frameNr
    # frameNr = re.search('\/.*()\/', list_entry)
    # if m:
    #     found = m.group(1)
    #
    # splitFP = os.path.split(list_entry)
    #
    # path = splitFP[0]
    # name = os.path.splitext(splitFP[1])[0]
    # extension = os.path.splitext(splitFP[1])[1]


    # print "List_entry[1]: " + str(list_entry[1])

    #break