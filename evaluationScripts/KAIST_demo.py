'''
    In this example, we will load a RefineDet model and use it to detect objects.

    USAGE: python KAIST_demo.py <GPU-ID for execution>

    ATTENTION:
        Please adapt paths in files "pascal_voc.py" before starting this script!
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
# Make sure that caffe is on the python path:
caffe_root = "{}/code/caffe/RefineDet".format(os.environ['HOME'])
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def ShowResults(img, image_file, results, labelmap, threshold=0.6, save_fig=False):
    plt.clf()
    plt.imshow(img)
    plt.axis('off')
    ax = plt.gca()

    num_classes = len(labelmap.item) - 1
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    for i in range(0, results.shape[0]):
        score = results[i, -2]
        if threshold and score < threshold:
            continue

        label = int(results[i, -1])
        name = get_labelname(labelmap, label)[0]
        color = colors[label % num_classes]

        xmin = int(round(results[i, 0]))
        ymin = int(round(results[i, 1]))
        xmax = int(round(results[i, 2]))
        ymax = int(round(results[i, 3]))
        coords = (xmin, ymin), xmax - xmin, ymax - ymin
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=3))
        display_text = '%s: %.2f' % (name, score)
        ax.text(xmin, ymin, display_text, bbox={'facecolor':color, 'alpha':0.5})
    if save_fig:
        plt.savefig(image_file[:-4] + '_dets.jpg', bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    ##### CONFIGURATION ################################################################################################

    # ***ATTENTION***: caffe_root defined in import section at beginning of this file!!!

    # Change "atWORK" to switch between HOME and WORK directories (False: HOME - True: WORK)
    atWork  = True

    file_postfix    = "iter_10000"  # Select which trained state to use for analysis
    dataset_name    = "KAIST"
    subsetName      = "train-all-T"
    job_name        = "refinedet_vgg16_320x320"

    det_threshold   = 0.5  # Detection threshold level to use

    # Path to labelmap file
    labelmap_file = "{}/code/temporalDetection_-EIGEN-/KAIST_preparation/labelmap_{}.prototxt".format(os.environ['HOME'], dataset_name)

    # Input path prefix
    path_prefix_HOME    = "{}/train_test_data".format(os.environ['HOME'])
    path_prefix_WORK    = "/net4/merkur/storage/deeplearning/users/gueste/TRAINING_test"
    ####################################################################################################################


    path_prefix = path_prefix_WORK if atWork else path_prefix_HOME
    assert os.path.exists(path_prefix), \
        "Path {} does not exist! --> atWORK = ?".format(path_prefix)


    kaist_path      = '{}/models/VGGNet/KAIST/{}/{}/'.format(path_prefix, subsetName, job_name)
    model_def       = os.path.join(kaist_path, 'deploy.prototxt')
    model_weights   = os.path.join(kaist_path, '{}_{}_{}.caffemodel'.format(dataset_name, job_name, file_postfix))

    # gpu preparation
    gpu_id = int(sys.argv[1])   #Adapted to use with script "StartIfGPUFree.py". GPU to use for execution

    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()

    # load labelmap
    #labelmap_file = 'data/VOC0712/labelmap_voc.prototxt'    # ORIGINAL
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    # load model
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # image preprocessing
    if '320' in model_def:
        img_resize = 320
    else:
        img_resize = 512
    net.blobs['data'].reshape(1, 3, img_resize, img_resize)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel                                               # Adapt!
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    # im_names = os.listdir('examples/images')
    im_names = ['000456.jpg', '000542.jpg', '001150.jpg', '001763.jpg', '004545.jpg', \
                'RGB_1.png', 'RGB_2.png', 'RGB_3.png', 'T_1.png', 'T_2.png', 'T_3.png']                                 # Takes .png images?
    for im_name in im_names:
        image_file = 'examples/images/' + im_name
        image = caffe.io.load_image(image_file)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        detections = net.forward()['detection_out']
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3] * image.shape[1]
        det_ymin = detections[0, 0, :, 4] * image.shape[0]
        det_xmax = detections[0, 0, :, 5] * image.shape[1]
        det_ymax = detections[0, 0, :, 6] * image.shape[0]
        result = np.column_stack([det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label])

        # show result
        #ShowResults(image, image_file, result, labelmap, 0.6, save_fig=False)  # ORIGINAL
        ShowResults(image, image_file, result, labelmap, threshold=det_threshold, save_fig=False)
