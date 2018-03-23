'''
    This module implements some useful functions which can be used in other scripts.
'''
import os
import sys



def fileParts(fileWithPath):
    '''
    This method takes the whole path of a file and returns ['path', 'fileName', 'fileExtension']
    :param fileWithPath: Complete path with file name and extension.
    :return: ['path', 'fileName', 'fileExtension'] as strings
    '''

    fileWithPath = str(fileWithPath)
    if len(fileWithPath) != 0:
        splitFP = os.path.split(fileWithPath)

        path        = splitFP[0]
        name        = os.path.splitext(splitFP[1])[0]
        extension   = os.path.splitext(splitFP[1])[1]

        if len(name) != 0:
            return [path, name, extension]
        else:
            print "ERROR: Empty file name!"

    else:
        print "WARNING: Given path is empty!"
