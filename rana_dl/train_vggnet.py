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
import warnings
from time import time

import mxnet as mx
from mxnet import gluon

from models import config
from models.networks.mxvggnet import VGG19

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-lr", "--learning-rate", type=float, default=1e-3,
                help="learning rate to use for training")
ap.add_argument("-n", "--num-devices", type=int, default=1,
                help="number of GPUs to run on")
ap.add_argument("-p", "--prefix", required=True,
                help="name of model prefix")
ap.add_argument("-m", "--means", required=True,
                help="path to image channels means file associated with training data")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
ap.add_argument("-e", "--end-epoch", type=int, default=100,
                help="epoch to end training at")
args = vars(ap.parse_args())

logging.basicConfig(level=logging.DEBUG,
                    filename=os.path.sep.join([config.LOG_OUTPUT, "vggnet",
                                               args["prefix"] + "_{}.log".format(args["start_epoch"])]),
                    filemode="w")


class DataIterLoader():
    def __init__(self, data_iter):
        self.data_iter = data_iter

    def __iter__(self):
        self.data_iter.reset()
        return self

    def __next__(self):
        batch = self.data_iter.__next__()
        assert len(batch.data) == len(batch.label) == 1
        data = batch.data[0]
        label = batch.label[0]
        return data, label

    def next(self):
        return self.__next__() # for Python 2


def forward_backward(net, data, label):
    with mx.autograd.record():
        outputs = [net(x) for x in data]
        losses = [loss_fn(X, Y) for X, Y in zip(outputs, label)]
    for l in losses:
        l.backward()

    return outputs, losses


def train_batch(data_it, label_it, ctx, net, trainer, metric):
    # Split the data batch and load them on GPUs
    data = gluon.utils.split_and_load(data_it, ctx)
    label = gluon.utils.split_and_load(label_it, ctx)
    # Compute gradient
    output, loss = forward_backward(net, data, label)
    # Update parameters
    trainer.step(data_it.shape[0])
    # Update metrics
    metric.update(label, output)
    # Return loss for summing
    return loss


def valid_batch(val_data, ctx, model):
    metric = mx.metric.Accuracy()
    for data, label in val_data:
        data = gluon.utils.split_and_load(data, ctx, even_split=False)
        label = gluon.utils.split_and_load(label, ctx, even_split=False)
        outputs = [model(x) for x in data]
        metric.update(label, outputs)

    return metric.get()


# Load the RGB means for the training set, then determine the batch
# size
means = json.loads(open(args["means"]).read())
bat_size = config.BATCH_SIZE * args["num_devices"]

train_iter = mx.io.ImageRecordIter(
    path_imgrec=config.TRAIN_MX_REC,
    data_shape=(3, 480, 640),
    batch_size=bat_size,
    rand_mirror=True,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"],
    preprocess_threads=args["num_devices"] * 2
)

val_iter = mx.io.ImageRecordIter(
    path_imgrec=config.VAL_MX_REC,
    data_shape=(3, 480, 640),
    batch_size=bat_size,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"]
)

train_iter_loader = DataIterLoader(train_iter)
val_iter_loader = DataIterLoader(val_iter)

num_batch = len(train_iter_loader)

# Construct the checkpoints path
checkpoints_path = os.path.sep.join([args["checkpoints"],
                                     args["prefix"]])

# If there is no specific model starting epoch supplied, then
# initialize the network
if args["start_epoch"] <= 0:
    # Build the VGGNet architecture
    print("[INFO] Building network...")
    model = VGG19()

# Otherwise, a specific checkpoint was supplied
else:
    # Load the checkpoint from disk
    print("[INFO] Loading epoch {}...".format(args["start_epoch"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Figure out checkpoint filename
        pad = 4 - len(str(args["start_epoch"]))
        zeroes = "0" * pad
        fname = args["prefix"] + "-" + zeroes + str(args["start_epoch"])
        # Load our model
        model = gluon.SymbolBlock.imports(args["prefix"] + "-symbol.json", ["data"], fname)

ctx = [mx.gpu(i) for i in range(0, args["num_devices"])]

model.initialize(mx.initializer.MSRAPrelu(), ctx=ctx)
model.hybridize()
trainer = gluon.Trainer(model.collect_params(), "sgd", {"learning_rate": args["learning_rate"]})

# Define our loss function
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

metric = mx.metric.Accuracy()

# Train the network
print("[INFO] Training network...")
for epoch in range(args["end_epoch"]):
    # Training Loop
    start = time()
    train_loss = 0
    metric.reset()

    for d, l in train_iter_loader:  # start of mini-batch
        train_loss += train_batch(d, l, ctx, model, trainer, metric)
        mx.nd.waitall()  # Wait until all computations are finished to benchmark the time

    _, train_acc = metric.get()
    train_loss /= num_batch
    print("[Epoch {}] Training Time = {:.1f} sec | Train-acc: {:.3f}, loss: {:.3f}".format(epoch, time() - start,
                                                                                           train_acc, train_loss))

    # Validation loop
    _, val_acc = valid_batch(val_iter_loader, ctx, model)
    print("\tValidation Accuracy = {:.2f}".format(val_acc))

    # Save a checkpoint
    path = os.path.sep.join([checkpoints_path, args["prefix"]])
    print("Saving checkpoint file {} to {}...".format(path, checkpoints_path))
    model.export(path, epoch=epoch)
