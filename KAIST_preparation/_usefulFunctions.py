'''
    This module implements some useful functions which can be used in other scripts.
'''
import os
import sys
import re
import logging

### Configurations ###
logging.basicConfig(format='%(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
######################


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
            logging.debug(str([path, name, extension]))
            return [path, name, extension]
        else:
            logging.error("ERROR: Empty file name!")

    else:
        logging.warning("WARNING: Given path is empty!")


def dirRecursive(dirToSearch, regexToUse):
    '''
        This method takes a path which has to be searched through and a regex which is used to find files inside it
        that match its pattern. A list of all matching files inside dirToSearch is returned.
        
        HINT: To find files with a specific extension, e.g. '.py', use a regex like '.*.py$' to search from
        the end of the filenames.
        :param dirToSearch: Root directory which has to be searched through
        :param regexToUse: Filter pattern to use
        :return: A list of all matching files inside dirToSearch
    '''
    filesInFolder = []
    pattern = re.compile(regexToUse, re.IGNORECASE)
    for root, dir, fileList in os.walk(dirToSearch):
        for file in fileList:
            logging.debug("--------------------------")
            logging.debug("File:   " + str(file))
            logging.debug("Match?: " + str(pattern.match(file)))
            if pattern.match(file):
                #Reconstruct full file path
                fileWithPath = os.path.join(root, file)
                # Add file to List
                filesInFolder.append(fileWithPath)

    return filesInFolder