import argparse
import json
import os

import mxnet as mx

from models import config

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

# Load the checkpoints from disk
print("[INFO] Loading model...")
checkpoints_path = os.path.sep.join([args["checkpoints"],
                                     args["prefix"]])
model = mx.module.Module.load(checkpoints_path, args["epoch"])


# Compile the model
model.bind(data_shapes=test_iter.provide_data,
           label_shapes=test_iter.provide_label)

# Make predictions on the testing data
print("[INFO] Predicting on test data...")
metrics = [mx.metric.Accuracy(), mx.metric.F1(average="micro")]
rank1 = model.score(test_iter, eval_metric=metrics)[0][1]

# Display the rank-1 accuracies
print("[INFO] rank-1 Accuracy: {:.2f}%".format(rank1 * 100))

# The F1 metric inherits from a class that tracks the number
# of false positives and negatives, allowing us to extract them
metric = metrics[1]
print("False Positives: {}\n" 
      "False Negatives: {}\n" 
      "True Positives: {}\n" 
      "True Negatives: {}\n"
      .format(metric.metrics.false_positives, metric.metrics.false_negatives,
              metric.metrics.true_positives, metric.metrics.true_negatives))
