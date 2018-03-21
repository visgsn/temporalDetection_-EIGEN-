import os
#import random


### *** HOME ***                                                                        # anpassen!
#kaistFolder = '/home/herrma/deepdata/users/herrma/datasets/KAIST/'
### *** WORK ***
kaistFolder =   '/net4/merkur/storage/deeplearning/users/gueste/data/KAIST/data-kaist/'
trainImgSub =   'train-all20/images'
testImgSub =    'test-all/images'

outputFolder =  '/net4/merkur/storage/deeplearning/users/gueste/data/KAIST/'


trainFolder = kaistFolder + trainImgSub
testFolder = kaistFolder + testImgSub

with open(outputFolder + 'AdaptedImageSets/Main/trainval.txt', 'w+') as trainValFile:
    for filename in os.listdir(trainFolder):    # List of all files in trainFolder
        if os.path.splitext(filename)[1] != '.db':
            trainValFile.write(trainFolder + '/' + os.path.splitext(filename)[0] + '.png ')
            trainValFile.write(outputFolder + 'Annotations/' + os.path.splitext(filename)[0] + '.xml')
            trainValFile.write("\n")
	
with open(outputFolder + 'AdaptedImageSets/Main/test.txt', 'w+') as testFile:
    for filename in os.listdir(testFolder):     # List of all files in testFolder
        if os.path.splitext(filename)[1] != '.db':
            testFile.write(testFolder + '/' + os.path.splitext(filename)[0] + '.png ')
            testFile.write(outputFolder + 'Annotations/' + os.path.splitext(filename)[0] + '.xml')
            testFile.write("\n")