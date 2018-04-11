# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

##### CONFIGURATION ####################################################################################################
caffe_root  = "{}/code/caffe/RefineDet".format(os.environ['HOME'])
this_dir = os.path.dirname(__file__)    # Is also the output directory for test results!
########################################################################################################################

assert os.path.exists(caffe_root), \
    "Path to caffe_root ({}) does not exist! --> Edit this file, if necessary.".format(caffe_root)

# Add caffe to PYTHONPATH
caffe_path = os.path.join(caffe_root, 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = os.path.join(this_dir, 'lib')
add_path(lib_path)
