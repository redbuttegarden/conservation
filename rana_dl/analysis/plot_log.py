import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--network", required=True,
                help="name of the network")
ap.add_argument("-d", "--dataset", required=True,
                help="name of the dataset")
args = vars(ap.parse_args())

# Define the paths to the training logs
logs = [
    (5, os.path.sep.join([os.pardir, os.pardir, os.pardir,
                           "data/output", args["network"], "training_0.log"])),
    (50, os.path.sep.join([os.pardir, os.pardir, os.pardir,
                           "data/output", args["network"], "training_5.log"])),
    (52, os.path.sep.join([os.pardir, os.pardir, os.pardir,
                           "data/output", args["network"], "model3_50.log"]))
]

# Initialize the list of train rank-1 and rank-5 accuracies, along
# with the training loss
train_rank_1, train_loss = [], []

# Initialize the list of validation rank-1 and rank-5 accuracies,
# along with the validation loss
val_rank_1, val_loss = [], []

# Loop over the training logs
for (i, (end_epoch, p)) in enumerate(logs):
    # Load the contents of the log file, then initialize the batch
    # lists for the training and validation data
    rows = open(p).read().strip()
    b_train_rank_1, b_train_loss = [], []
    b_val_rank_1, b_val_loss = [], []

    # Grab the set of training epochs
    epochs = set(re.findall(r'Epoch\[(\d+)\]', rows))
    epochs = sorted([int(e) for e in epochs])

    # Loop over the epochs
    for e in epochs:
        # Find all rank-1 accuracies and loss
        # values, then take the final entry in the list for each
        s = r'Epoch\[' + str(e) + '\].*accuracy=([0]*\.?[0-9]+)'
        rank_1 = re.findall(s, rows)[0]
        s = r'Epoch\[' + str(e) + '\].*cross-entropy=([0-9]*\.?[0-9]+)'
        loss = re.findall(s, rows)[0]

        # Update the batch training lists
        b_train_rank_1.append(float(rank_1))
        b_train_loss.append(float(loss))

    # Extract the validation rank-1 for each
    # epoch, followed by the loss
    b_val_rank_1 = re.findall(r'Validation-accuracy=(.*)', rows)
    b_val_loss = re.findall(r'Validation-cross-entropy=(.*)', rows)

    # Convert the validation rank-1 and loss lists to floats
    b_val_rank_1 = [float(x) for x in b_val_rank_1]
    b_val_loss = [float(x) for x in b_val_loss]

    # Check to see if we are examining a log file other than the
    # first one, and if so, use the number of the final epoch in
    # the log file as our slice index
    if i > 0 and end_epoch is not None:
        train_end = end_epoch - logs[i - 1][0]
        val_end = end_epoch - logs[i - 1][0]

    # Otherwise, this is the first epoch so no subtraction needed
    else:
        train_end = end_epoch
        val_end = end_epoch

    # Update the training lists
    train_rank_1.extend(b_train_rank_1[0:train_end])
    train_loss.extend(b_train_loss[0:train_end])

    # Update the validation lists
    val_rank_1.extend(b_val_rank_1[0:val_end])
    val_loss.extend(b_val_loss[0:val_end])

# Plot the accuracies
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(train_rank_1)), train_rank_1,
         label="train_rank_1")
plt.plot(np.arange(0, len(val_rank_1)), val_rank_1,
         label="val_rank_1")
plt.title("{}: Rank-1 accuracy on {}".format(
    args["network"], args["dataset"]
))
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")

# Plot the losses
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(train_loss)), train_loss,
         label="train_loss")
plt.plot(np.arange(0, len(val_loss)), val_loss,
         label="val_loss")
plt.title("{}: Cross-entropy loss on {}".format(args["network"],
                                                args["dataset"]))
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()
