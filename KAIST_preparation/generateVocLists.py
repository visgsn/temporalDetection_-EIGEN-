import os
#import random


### *** HOME ***                                                                        # anpassen!
#kaistFolder = '/home/herrma/deepdata/users/herrma/datasets/KAIST/'
### *** WORK ***
kaistFolder = '/net4/merkur/storage/deeplearning/users/gueste/data/KAIST/'#data-kaist/' ???


trainFolder = kaistFolder + 'JPEGImagesTrain'                                           # train20?
testFolder = kaistFolder + 'JPEGImagesTest'                                             # test?

with open(kaistFolder + 'ImageSets/Main/trainval.txt', 'w+') as trainValFile:
    for filename in os.listdir(trainFolder):    # List of all files in trainFolder      # Ordner existiert nicht?!?
        if os.path.splitext(filename)[1] != '.db':
            trainValFile.write('JPEGImages/' + os.path.splitext(filename)[0] + '.png ')
            trainValFile.write('Annotations/' + os.path.splitext(filename)[0] + '.xml')
            trainValFile.write("\n")
	
with open(kaistFolder + 'ImageSets/Main/test.txt', 'w+') as testFile:
    for filename in os.listdir(testFolder):                                             # Ordner existiert nicht?!?
        if os.path.splitext(filename)[1] != '.db':
            testFile.write('JPEGImages/' + os.path.splitext(filename)[0] + '.png ')
            testFile.write('Annotations/' + os.path.splitext(filename)[0] + '.xml')
            testFile.write("\n")