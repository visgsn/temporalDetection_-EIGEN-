'''
    This script is used to copy and rename the images and annotations from the KAIST dataset, which have previously
    been extracted with Pjotr's Matlab Toolbox. (By default only training data from set00 to set05)
    Names are chosen according to the scheme of "train_all20" with one exception - instead of the unique ID at the
    end of the generated file names, the last element of 'outputDir' is used as postfix (e.g. _train-all).
    Example for images:
        T_tmp_set00_V000_I00019_train-all.png         (T_ stands for "thermal", RGB_ for color images)
    Example for annotations:
        set00_V000_I00019_train-all.txt
'''

from _usefulFunctions import *
import shutil
import os
import sys
import re
import logging

##### Configurations ########################################
### *** HOME ***
annoDir     = "/home/gueste/data/KAIST/data-kaist/annotations-extracted"    # Input annotations
imageDir    = "/home/gueste/data/KAIST/data-kaist/videos-extracted"         # Input images
outputDir   = "/home/gueste/data/KAIST/data-kaist/train-all-T"              # Last element of path used as data postfix!
### *** WORK ***
# Insert config for WORK here...

extractThermal  = True              # If true, thermal images will be extracted into outputDir
extractRgb      = False             # If true, RGB images will be extracted into outputDir

annoRegex       = '.*.txt$'         # Used to find available annotations
imageRegex      = 'frame_.*.png$'   # Used to find available images
extractSetRegex = '.*\/set0[0-5]\/' # Specifies which sets to extract (Default: '.*\/set0[0-5]\/')
splitThermRegex = '.*\/LWIR_V'      # Used to separate RGB- and thermal images
logging.basicConfig(format='%(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
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

    postfix = os.path.split(outputDir)[1]

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

    # Verify correct image file order (lined up with annotations)
    if not( re.search(frameNr, thermalImages[i]) and re.search(subset, thermalImages[i]) and \
            re.search(set, thermalImages[i]) and re.search(frameNr, rgbImages[i]) and \
            re.search(subset, rgbImages[i]) and re.search(set, rgbImages[i]) ):
        logging.error("ERROR: Annotation file order does not match Image file order!")
        sys.exit(1)

    # Wrap single parts together to one big filename for each filetype
    nameAnno_tmp    = "{}_{}_I{}_{}{}".format(set, subset, frameNr, postfix, extensionAnno)
    nameThermal_tmp = "T_tmp_{}_{}_I{}_{}{}".format(set, subset, frameNr, postfix, extensionTherm)
    nameRgb_tmp     = "RGB_tmp_{}_{}_I{}_{}{}".format(set, subset, frameNr, postfix, extensionRgb)
    # Append names to lists
    nameAnno.append(nameAnno_tmp)
    nameThermal.append(nameThermal_tmp)
    nameRgb.append(nameRgb_tmp)
    logging.debug("nameAnno     : " + str(nameAnno[i]))
    logging.debug("nameThermal  : " + str(nameThermal[i]))
    logging.debug("nameRgb      : " + str(nameRgb[i]))
    logging.debug("----------------------------------------------------------")


# Copy files to new destination folder (outputDir/...)
numImages = len(nameAnno)
if extractThermal:
    logging.info("Thermal images will be copied")
    numImages += len(nameThermal)
if extractRgb:
    logging.info("RGB images will be copied")
    numImages += len(nameRgb)
logging.info("Start copying " + str(numImages) + " files to output folders:\n'" \
             + annoDir_out + "' and\n'" \
             + imageDir_out + "'\n")
progressOnePercent  = len(nameAnno) / 100 # Used to display progress while copying
currentProgress     = 0
for i in range(len(availableAnnos)):
    # Construct full paths
    annoFile_out    = os.path.join(annoDir_out, nameAnno[i])
    thermalFile_out = os.path.join(imageDir_out, nameThermal[i])
    rgbFile_out     = os.path.join(imageDir_out, nameRgb[i])
    #Copy file if not existing
    if not os.path.isfile(annoFile_out):
        shutil.copyfile(availableAnnos[i], annoFile_out)
    else: logging.warning("WARNING: File '" + str(annoFile_out) + "' already esisting. --> NOT copied!")

    if extractThermal:
        if not os.path.isfile(thermalFile_out):
            shutil.copyfile(thermalImages[i], thermalFile_out)
        else: logging.warning("WARNING: File '" + str(thermalFile_out) + "' already esisting. --> NOT copied!")

    if extractRgb:
        if not os.path.isfile(rgbFile_out):
            shutil.copyfile(rgbImages[i], rgbFile_out)
        else: logging.warning("WARNING: File '" + str(rgbFile_out) + "' already esisting. --> NOT copied!")

    # Print progress status
    if 0 == i % progressOnePercent:
        logging.info(str(currentProgress) + "% complete")
        currentProgress += 1

print "\n\n"
logging.info("DONE.")
