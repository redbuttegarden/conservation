import argparse
import json
import os
import warnings

import mxnet as mx
from mxnet import gluon

from models import config
from utils.data_iter_loader import DataIterLoader

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-m", "--means", required=True,
                help="path to image channels means file associated with training data")
ap.add_argument("-p", "--prefix", required=True,
                help="name of model prefix")
ap.add_argument("-r", "--record", required=True,
                help="path to record file to be used for testing")
ap.add_argument("-e", "--epoch", type=int, required=True,
                help="epoch number to load")
args = vars(ap.parse_args())

# Load the RGB means for the training set
means = json.loads(open(args["means"]).read())

# Construct the testing image iterator
test_iter = mx.io.ImageRecordIter(
    path_imgrec=args["record"],
    data_shape=(3, 480, 640),
    batch_size=config.BATCH_SIZE,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"]
)

test_iter_loader = DataIterLoader(test_iter)

# Load the checkpoint from disk
print("[INFO] Loading epoch {}...".format(args["epoch"]))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Figure out checkpoint filename
    pad = 4 - len(str(args["epoch"]))
    zeroes = "0" * pad
    fname = args["prefix"] + "-" + zeroes + str(args["epoch"]) + ".params"
    model = gluon.SymbolBlock.imports(os.path.sep.join([args["checkpoints"], args["prefix"] + "-symbol.json"]),
                                      ["data"], os.path.sep.join([args["checkpoints"], fname]))

# Make predictions on the testing data
print("[INFO] Predicting on test data...")
metrics = [mx.metric.Accuracy(), mx.metric.F1(average="micro")]
for data, label in test_iter_loader:
    data = data.as_in_context(mx.cpu())
    label = label.as_in_context(mx.cpu())
    output = model(data)
    for metric in metrics:
        metric.update(label, output)

name, acc = metrics[0].get()
print("Test {}={:.4f}%".format(name, acc * 100))

# The F1 metric inherits from a class that tracks the number
# of false positives and negatives, allowing us to extract them
metric = metrics[1]
print("False Positives: {}\n" 
      "False Negatives: {}\n" 
      "True Positives: {}\n" 
      "True Negatives: {}\n"
      .format(metric.metrics.false_positives, metric.metrics.false_negatives,
              metric.metrics.true_positives, metric.metrics.true_negatives))
