# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import matlab.engine
import os
from datasets._usefulFunctions import fileParts
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from kaist_eval import kaist_eval
from fast_rcnn.config import cfg

class kaist(imdb):
    def __init__(self, image_set, name, kaist_path=None, kaist_classes=None):
        imdb.__init__(self, 'kaist_' + name + '_' + image_set)
        self._name = name
        self._image_set = image_set

        ##### CONFIGURATION ############################################################################################
        if kaist_path == None:
            ### *** HOME ***
            #self._kaist_path = '{}/data/KAIST'.format(os.environ['HOME'])                                               # kaist_path usually given from 'factory.py'
            ### *** WORK ***
            self._kaist_path = '/net4/merkur/storage/deeplearning/users/gueste/data/KAIST'                              # kaist_path usually given from 'factory.py'
        ################################################################################################################
        else:
            self._kaist_path = kaist_path

        self._ImageSets_path    = os.path.join(self._kaist_path, self._name)
        #self._ImageData_path    = os.path.join(self._kaist_path, 'data-kaist', 'test-all')
        self._ImageData_path    = None
        self._Annotations_path  = None
        #self._classes = ('__background__', # always index 0
        #                 'person')
        self._classes = kaist_classes if not kaist_classes == None else ('__background__', 'person')                    # Usually given from 'factory.py'
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._use_07_metric = False     # Shoult be 'False' in order to ensure correct calculation of Average Precision
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : True,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._kaist_path), \
                'KAIST path does not exist: {}'.format(self._kaist_path)
        assert os.path.exists(self._ImageSets_path), \
                'Path does not exist: {}'.format(self._ImageSets_path)
        assert os.path.exists(self._ImageData_path), \
                'Path does not exist: {}'.format(self._ImageData_path)
        assert os.path.exists(self._Annotations_path), \
                'Path does not exist: {}'.format(self._Annotations_path)

    def image_path_at(self, i):                                                                                         # CHECKED!
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, indexOrName):                                                                       # CHECKED!
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._ImageData_path, indexOrName + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):                                                                                    # CHECKED!
        """
        Load the indexes listed in this dataset's image set file.
        And extracts the paths for the image data (_ImageData_path) and the
        annotations (_Annotations_path) out of the image_set_file
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._ImageSets_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        pathsExtracted = False
        with open(image_set_file) as f:
            image_index = []
            for line in f:
                image_index.append(fileParts(line.split()[0])[1])
                if not pathsExtracted:
                    self._ImageData_path    = os.path.join(self._kaist_path, fileParts(line.split()[0])[0])
                    self._Annotations_path  = os.path.join(self._kaist_path, fileParts(line.split()[1])[0])
                    pathsExtracted = True
        return image_index

    def _get_default_path(self):                                                                                        # CHECKED!
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join('path to pascal', 'KAIST')

    def gt_roidb(self):                                                                                                 # CHECKED!
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]                                      # Strange: --> _image_index ?!?
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):                                                                                   # CHECKED!
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._name == '2007' or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):                                                                                                # CHECKED!
        if self._name == '2007' or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)                      # Strange: Fkt. not available!?!

    def _load_selective_search_roidb(self, gt_roidb):                                                                   # CHECKED!
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))                            # Strange: --> _name ?!?
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)                      # Strange: Fkt. not available!?!

    def _load_pascal_annotation(self, index):                                                                           # CHECKED!
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._Annotations_path, index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):                                                                                             # CHECKED!
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):                                                                           # CHECKED!
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        if self._name == "2012":
            path = os.path.join(
                self._kaist_path,
                cfg.net_name,
                'results',
                'KAIST_' + self._name,
                'Main',
                filename)
        else:
            path = os.path.join(
                self._kaist_path,
                'results',
                'KAIST_' + self._name,
                'Main',
                filename)
        return path

    def _get_kaist_results_file_template(self, output_dir):                                                             # SELF WRITTEN!
        # Example path:
        # ~/code/temporalDetection_-EIGEN-/evaluationScripts/output/default/train-all-T/...
        # KAIST_refinedet_it50184_320x320_iter_2500/detections_for_matlab/<comp_id>_det_test_person.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(output_dir, "detections_for_matlab", filename)
        return path

    def _write_voc_results_file(self, all_boxes):                                                                       # CHECKED!
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} KAIST results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:f} {:f} {:f} {:f} {:f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _write_kaist_results_file(self, all_boxes, output_dir):                                                         # SELF WRITTEN!
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} KAIST results file'.format(cls)
            filename = self._get_kaist_results_file_template(output_dir).format(cls)
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:d} {:f} {:f} {:f} {:f} {:f}\n'.
                                format(im_ind + 1,                          # index
                                       (dets[k, 0] + 1),                    # left
                                       (dets[k, 1] + 1),                    # top
                                       (dets[k, 2] + 1) - (dets[k, 0] + 1), # width
                                       (dets[k, 3] + 1) - (dets[k, 1] + 1), # height
                                       dets[k, -1] * 100))                  # score

    def _do_python_eval(self, output_dir = 'output'):                                                                   # CHECKED!
        annopath = os.path.join(
            self._Annotations_path,
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._ImageSets_path,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._kaist_path, 'annotations_cache')
        aps     = []
        precs   = []
        recs    = []
        # The PASCAL VOC metric changed in 2010
        #use_07_metric = True if int(self._name) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if self._use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = kaist_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=self._use_07_metric)
            aps     += [ap]
            precs   += [np.mean(prec)]
            recs    += [np.mean(rec)]
            print('AP, Precision, Recall for {} = {:.4f},  {:.4f},  {:.4f}'
                  .format(cls, ap, np.mean(prec), np.mean(rec)))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean (AP, Precision, Recall) = ({:.4f}, {:.4f}, {:.4f})'
              .format(np.mean(aps), np.mean(precs), np.mean(recs)))
        print('~~~~~~~~')
        print('Results (mAP, Precision, Recall):')
        for i, ap in enumerate(aps):
            print('{:.3f},  {:.3f},  {:.3f}'.format(ap, precs[i], recs[i]))
        print('\nMean: {:.3f},  {:.3f},  {:.3f}'.format(np.mean(aps), np.mean(precs), np.mean(recs)))
        print('~~~~~~~~')
        cfg.mAP = np.mean(aps)
        cfg.prec = np.mean(precs)
        cfg.rec = np.mean(recs)

    def _do_matlab_eval(self, output_dir='output'):                                                                     # SELF WRITTEN!!!
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        # Start MATLAB engineand add toolbox path
        mateng = matlab.engine.start_matlab()
        mateng.addpath(mateng.genpath('~/code/temporalDetection_-EIGEN-/piotr-toolbox-3.40/'))

        # Reconsruct detection file name for class 'person', dir to store the results and the KAIST data dir
        detectionFile = self._get_kaist_results_file_template(output_dir).format('person')
        result_dir = os.path.split(detectionFile)[0] + '/'
        data_dir = os.path.join(self._kaist_path, 'data-kaist/')
        print("Using the following paths for MATLAB evaluation:")
        print("Matlab detectionFile: " + str(detectionFile))
        print("Matlab resultDir    : " + str(result_dir))
        print("Matlab dataDir      : " + str(data_dir))

        # Call MATLAB evaluation script
        mateng.evalKAIST(result_dir, data_dir, detectionFile)
        print("done")

    def evaluate_detections(self, all_boxes, output_dir):                                                               # CHECKED!
        self._write_voc_results_file(all_boxes)
        self._write_kaist_results_file(all_boxes, output_dir)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):                                                                                     # CHECKED!
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.kaist import kaist
    d = kaist('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
