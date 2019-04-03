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

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision.datasets import ImageRecordDataset

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
# ap.add_argument("-m", "--means", required=True,
#                 help="path to image channels means file associated with training data")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
ap.add_argument("-e", "--end-epoch", type=int, default=100,
                help="epoch to end training at")
args = vars(ap.parse_args())

logging.basicConfig(level=logging.DEBUG,
                    filename=os.path.sep.join([config.LOG_OUTPUT, "vggnet",
                                               args["prefix"] + "_{}.log".format(args["start_epoch"])]),
                    filemode="w")

# Load the RGB means for the training set, then determine the batch
# size
# means = json.loads(open(args["means"]).read())
bat_size = config.BATCH_SIZE * args["num_devices"]

train_dataset = ImageRecordDataset(config.TRAIN_MX_REC)
transformer = gluon.data.vision.transforms.ToTensor()
train_dataset = train_dataset.transform(transformer)

val_dataset = ImageRecordDataset(config.VAL_MX_REC)
val_dataset = val_dataset.transform(transformer)

train_dataloader = DataLoader(train_dataset, batch_size=bat_size, shuffle=True,
                              num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=bat_size, shuffle=False,
                            num_workers=4)

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
trainer = gluon.Trainer(model.collect_params(), "sgd", {"learning_date": args["learning_rate"]})

# Initialize the evaluation metrics
metrics = [mx.metric.Accuracy()]

# Define our loss function
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

# Train the network
print("[INFO] Training network...")
for epoch in range(args["end_epoch"]):
    # Training Loop
    cumulative_train_loss = mx.nd.zeros(1, ctx=ctx)
    training_samples = 0

    for batch_idx, (data, label) in enumerate(train_dataloader):  # start of mini-batch
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        with mx.autograd.record():
            output = model(data)  # Forward pass
            loss = loss_fn(output, label)  # Get loss

        loss.backward()  # Compute gradients
        trainer.step(data.shape[0])  # Update weights with SGD
        cumulative_train_loss += loss.sum()
        training_samples += data.shape[0]
        metrics[0].update(label, output)  # Update the metrics # end of mini-batch

    train_loss = cumulative_train_loss.asscalar() / training_samples

    # Validation loop
    cumulative_val_loss = mx.nd.zeros(1, ctx)
    val_samples = 0
    for batch_idx, (data, label) in enumerate(val_dataloader):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        output = model(data)  # Forward pass
        loss = loss_fn(output, label)  # Get loss

        cumulative_val_loss += loss.sum()
        val_samples += data.shape[0]

    val_loss = cumulative_val_loss.asscalar() / val_samples

    name, acc = metrics[0].get()
    print("[Epoch {}] Training metrics: {}={}".format(epoch, name, acc))
    print("[Epoch {}] Training loss: {:.2f}, validation loss: {:.2f}".format(epoch, train_loss, val_loss))
    metrics[0].reset()  # End of epoch

    # Save a checkpoint
    path = os.path.sep.join([checkpoints_path, args["prefix"]])
    print("Saving checkpoint file {} to {}...".format(path, checkpoints_path))
    model.export(path, epoch=epoch)
