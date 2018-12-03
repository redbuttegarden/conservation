import argparse
import json
import os

import mxnet as mx

from models import config

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True,
                help="name of model prefix")
ap.add_argument("-e", "--epoch", type=int, required=True,
                help="epoch number to load")
args = vars(ap.parse_args())

# Load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# Construct the testing image iterator
test_iter = mx.io.ImageRecordIter(
    path_imgrec=config.TEST_MX_REC,
    data_shape=(3, 84, 84),
    batch_size=config.BATCH_SIZE,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"]
)

# Load the checkpoints from disk
print("[INFO] Loading model...")
checkpoints_path = os.path.sep.join([args["checkpoints"],
                                     args["prefix"]])
model = mx.model.FeedForward.load(checkpoints_path,
                                  args["epoch"])

# Compile the model
model = mx.model.FeedForward(
    ctx=[mx.gpu(0)],
    symbol=model.symbol,
    arg_params=model.arg_params,
    aux_params=model.aux_params
)

# Make predictions on the testing data
print("[INFO] Predicting on test data...")
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5)]
(rank1, rank5) = model.score(test_iter, eval_metric=metrics)

# Display the rank-1 and rank-5 accuracies
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))
