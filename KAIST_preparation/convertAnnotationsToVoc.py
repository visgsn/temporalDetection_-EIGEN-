'''
    This script converts all annotations from the KAIST dataset to a compatible format for VOC
    and exports them as .xml files into the output folder (renamed according to image name).

    Afterwards you can run "generateVocLists.py"

    Check paths below before execution!
'''

from _usefulFunctions import *
import os
import sys
import logging


##### Configurations ########################################
### *** HOME ***
kaistFolder =   '/home/gueste/data/KAIST/data-kaist'
### *** WORK ***
#kaistFolder =   '/net4/merkur/storage/deeplearning/users/gueste/data/KAIST/data-kaist/'

dataToExtract   = ['test-all', 'train-all-T']
excludeLabels   = ['people', 'person?', 'cyclist']
useThermal      = True  # If False, 'RGB_'-images will be extracted.

logging.basicConfig(format='%(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
#############################################################


outFolder = os.path.join(os.path.split(kaistFolder)[0], 'Annotations/')

if not os.path.isdir(outFolder):
    logging.info("Creating output directory '" + str(outFolder) + "' because it doesn't exist.")
    os.makedirs(outFolder)

for folder in dataToExtract:
    annoFiles = dirRecursive(os.path.join(kaistFolder, folder, 'annotations'), '.*.txt$')
    if useThermal:
        imgFiles = dirRecursive(os.path.join(kaistFolder, folder, 'images'), 'T_.*.png$')   # "T_" --> Thermal imag.
    else:
        imgFiles = dirRecursive(os.path.join(kaistFolder, folder, 'images'), 'RGB_.*.png$') # "RGB_" --> Colored imag.

    for (i,fPath) in enumerate(annoFiles):
        pa,imgName,ex = fileParts(imgFiles[i])

        # Check if imgFile name contains whitespaces (If true: Rename file!)
        if " " in imgName:
            imgName_new = imgName.replace(' ', '')
            logging.info("File name contains whitespaces ('" + str(imgFiles[i]) + "') --> renaming file!\n" + \
                         "OLD: " + str(imgFiles[i]) + "\n" + \
                         "NEW: " + str(os.path.join(pa, imgName_new + ex)))
            os.rename(imgFiles[i], os.path.join(pa, imgName_new + ex))
            imgName = imgName_new[:]
        # Check if path contains any whitespaces (Abort if true!)
        elif " " in imgFiles[i] or " " in fPath:
            logging.error("ERROR: Path to File contains whitespace(s). Please remove and rerun this script!" + \
                         "(" + str(imgFiles[i]) + "  OR  " + str(fPath) + ")")
            sys.exit()

        with open(fPath) as f:

            logging.debug("Generating file '" + str(outFolder + imgName + '.xml') + "'")
            with open(outFolder + imgName + '.xml', 'w') as outFile:
                # write header
                outFile.write('<annotation>\n\t<folder>KAIST</folder>\n')
                outFile.write('\t<filename>' + imgName + '.png' + '</filename>\n')
                outFile.write('\t<source>\n\t\t<database>KAIST</database>\n')
                outFile.write('\t\t<annotation>KAIST Annotation</annotation>\n')
                outFile.write('\t\t<image>' + imgName + '</image>\n\t</source>\n')
                outFile.write('\t<size>\n\t\t<width>640</width>\n')
                outFile.write('\t\t<height>512</height>\n')
                outFile.write('\t\t<depth>3</depth>\n\t</size>\n')  # Dimensions correct? (depth)
                outFile.write('\t<segmented>0</segmented>\n')

                # write bounding boxes
                for line in f:
                    if '%' not in line:
                        parts = line.split(' ')

                        if parts[0] not in excludeLabels:
                            outFile.write('\t<object>\n\t\t')
                            outFile.write('<name>' + parts[0] + '</name>\n\t\t')
                            outFile.write('<bndbox>\n\t\t\t')
                            outFile.write('<xmin>' + parts[1] + '</xmin>\n\t\t\t')
                            outFile.write('<ymin>' + parts[2] + '</ymin>\n\t\t\t')
                            outFile.write('<xmax>{:d}</xmax>\n\t\t\t'.format(int(parts[1]) + int(parts[3]) - 1))
                            outFile.write('<ymax>{:d}</ymax>\n\t\t\t'.format(int(parts[2]) + int(parts[4]) - 1))
                            outFile.write('</bndbox>\n\t\t')
                            outFile.write('<difficult>0</difficult>\n\t')
                            outFile.write('</object>\n')

                outFile.write('</annotation>')
