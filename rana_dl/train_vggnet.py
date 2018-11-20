"""
VGG19 Implementation based on:
Karen Simonyan and Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition". In: CoRR
abs/1409.1556 (2014). URL: http://arxiv.org/abs/1409.1556

Code adapted from:
Rosebrock, Adrian. “Training VGGNet on ImageNet.” Deep Learning for Computer Vision with Python, 1.3 ed., PyImageSearch,
2017.
"""
import argparse
import json
import logging
import os
import random

from imutils import paths
import mxnet as mx

from rana_dl.models.networks.mxvggnet import MxVGGNet

NUM_CLASSES = 2
MX_OUTPUT = os.path.expanduser("~/data")
TRAIN_MX_LIST = os.path.sep.join([MX_OUTPUT, "lists/train.lst"])
VAL_MX_LIST = os.path.sep.join([MX_OUTPUT, "lists/val.lst"])
TEST_MX_LIST = os.path.sep.join([MX_OUTPUT, "lists/test.lst"])

TRAIN_MX_REC = os.path.sep.join([MX_OUTPUT, "rec/train.rec"])
VAL_MX_REC = os.path.sep.join([MX_OUTPUT, "rec/val.rec"])
TEST_MX_REC = os.path.sep.join([MX_OUTPUT, "rec/test.rec"])

DATASET_MEAN = "output/pollinator_mean.json"

BATCH_SIZE = 64
NUM_DEVICES = 8

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True,
                help="name of model prefix")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
args = vars(ap.parse_args())

logging.basicConfig(level=logging.DEBUG,
                    filename="training_{}.log".format(args["start_epoch"]),
                    filemode="w")

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
image_paths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(image_paths)

# Loop over the input images
for img_path in image_paths:
    # Load the image, pre-process it, and store it in the data list
    image = cv2.imread(img_path)
    image = cv2.resize(image, (84, 84))
    image = img_to_array(image)
    data.append(image)

    # Extract the class label from the image path and update the
    # labels list
    label = img_path.split(os.path.sep)[-2]
    label = 1 if label == "Pollinator" else 0
    labels.append(label)

"""
At this point labels are imbalanced so we want to figure out the
most frequent label and remove data with that label until both
label classes are evenly represented.
"""
c = Counter(labels)
label_freqs = c.most_common(2)
over_represented, or_freq = label_freqs[0]
under_represented, ur_freq = label_freqs[1]

# Now that we have the label frequencies, we purge the
# over-represented data
paired = zip(labels, data)
sorted_paired = sorted(paired, key=lambda x: x[0])  # Sort by the label
# Unzip a subset of the data based on the difference between labels
labels, data = zip(*sorted_paired[or_freq - ur_freq:])

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
