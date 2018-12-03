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

import mxnet as mx

from models import config
from models.networks.mxvggnet import MxVGGNet

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True,
                help="name of model prefix")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
args = vars(ap.parse_args())

logging.basicConfig(level=logging.DEBUG,
                    filename=os.path.sep.join([config.MX_OUTPUT, "training_{}".format(args["start_epoch"])]),
                    filemode="w")

# Load the RGB means for the training set, then determine the batch
# size
means = json.loads(open(config.DATASET_MEAN).read())
bat_size = config.BATCH_SIZE * config.NUM_DEVICES

train_iter = mx.io.ImageRecordIter(
    path_imgrec=config.TRAIN_MX_REC,
    data_shape=(3, 84, 84),
    batch_size=bat_size,
    rand_crop=True,
    rand_mirror=True,
    rotate=15,
    max_shear_ratio=0.1,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"],
    preprocess_threads=config.NUM_DEVICES * 2
)

val_iter = mx.io.ImageRecordIter(
    path_imgrec=config.VAL_MX_REC,
    data_shape=(3, 84, 84),
    batch_size=bat_size,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"]
)

# Initialize the optimizer
opt = mx.optimizer.SGD(learning_rate=1e-2, momentum=0.9, wd=0.0005,
                       rescale_grad=1.0 / bat_size)

# Construct the checkpoints path, initialize the model argument and
# auxillary parameters
checkpoints_path = os.path.sep.join([args["checkpoints"],
                                     args["prefix"]])
arg_params = None
aux_params = None

# If there is no specific model starting epoch supplied, then
# initialize the network
if args["start_epoch"] <= 0:
    # Build the VGGNet architecture
    print("[INFO] Building network...")
    model = MxVGGNet.build(config.NUM_CLASSES)

# Otherwise, a specific checkpoint was supplied
else:
    # Load the checkpoint from disk
    print("[INFO] Loading epoch {}...".format(args["start_epoch"]))
    model = mx.model.FeedForward.load(checkpoints_path,
                                      args["start_epoch"])

    # Update the model and parameters
    arg_params = model.arg_params
    aux_params = model.aux_params
    model = model.symbol

# Compile the model
model = mx.model.FeedForward(
    ctx=[mx.gpu(i) for i in range(0, config.NUM_DEVICES)],
    symbol=model,
    initializer=mx.initializer.MSRAPrelu(),
    arg_params=arg_params,
    aux_params=aux_params,
    optimizer=opt,
    num_epoch=80,
    begin_epoch=args["start_epoch"]
)

# Initialize the callbacks and evaluation metrics
batch_end_CBs = [mx.callback.Speedometer(bat_size, 250)]
epoch_end_CBs = [mx.callback.do_checkpoint(checkpoints_path)]
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5),
           mx.metric.CrossEntropy()]

# Train the network
print("[INFO] Training network...")
model.fit(
    X=train_iter,
    eval_data=val_iter,
    eval_metric=metrics,
    batch_end_callback=batch_end_CBs,
    epoch_end_callback=epoch_end_CBs
)
