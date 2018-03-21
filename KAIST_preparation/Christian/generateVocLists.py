import os
#import random

kaistFolder = '/home/herrma/deepdata/users/herrma/datasets/KAIST/'
trainFolder = kaistFolder + 'JPEGImagesTrain'
testFolder = kaistFolder + 'JPEGImagesTest'

with open(kaistFolder + 'ImageSets/Main/trainval.txt', 'w+') as trainValFile:
    for filename in os.listdir(trainFolder):
        if os.path.splitext(filename)[1] != '.db':
            trainValFile.write('JPEGImages/' + os.path.splitext(filename)[0] + '.png ')
            trainValFile.write('Annotations/' + os.path.splitext(filename)[0] + '.xml')
            trainValFile.write("\n")
	
with open(kaistFolder + 'ImageSets/Main/test.txt', 'w+') as testFile:
    for filename in os.listdir(testFolder):
        if os.path.splitext(filename)[1] != '.db':
            testFile.write('JPEGImages/' + os.path.splitext(filename)[0] + '.png ')
            testFile.write('Annotations/' + os.path.splitext(filename)[0] + '.xml')
            testFile.write("\n")