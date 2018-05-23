# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Factory method for easily getting imdbs by name.
Update this file and KAIST_py if a new dataset should be used.
"""

__sets = {}

from datasets.kaist import kaist
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
import numpy as np
import os


##### CONFIGURATION ####################################################################################################
# Change "atWORK" to switch between HOME and WORK directories (False: HOME - True: WORK)
atWork  = True

# List of all (converted) KAIST-subsets - Append to list if new ones are available!
subsetsFromFile = True  # Whether to use subset list from file or from list below (kaist_subsets)
kaist_subsets   = ['train-all-T']
kaist_classes   = ('__background__',  # always index 0
                   'person')

subsetFileName  = 'SubsetList_factory.txt'
kaist_path_HOME = '{}/data/KAIST'.format(os.environ['HOME'])
kaist_path_WORK = '/net4/merkur/storage/deeplearning/users/gueste/data/KAIST'
########################################################################################################################


def _read_subsets_from_file():
    # Set and check file path
    subsetFilePath = os.path.join(kaist_path, subsetFileName)  # File has to be located in kaist_path
    assert os.path.isfile(subsetFilePath), \
        "File {} does not exist! --> Check path or create file".format(subsetFilePath)
    # Read file and extract subset names
    with open(subsetFilePath) as f:
        subsets = [singleSub.rstrip('\n') for singleSub in f]  # Subsets have to be separated by newlines!
    return subsets


# Set and check paths
kaist_path = kaist_path_WORK if atWork else kaist_path_HOME
assert os.path.exists(kaist_path), \
    "Path {} does not exist! --> atWORK = ?".format(kaist_path)
# Read subsets from file?!
if subsetsFromFile: kaist_subsets = _read_subsets_from_file()


# Set up KAIST_<subset_name>_<split> using selective search "fast" mode
for subset_name in kaist_subsets:
    for split in ['trainval', 'test']:
        name = 'kaist_{}_{}'.format(subset_name, split)
        __sets[name] = (lambda split=split, subset_name=subset_name, kaist_path=kaist_path: \
                            kaist(split, subset_name, kaist_path=kaist_path, kaist_classes=kaist_classes))

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012', '0712']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
