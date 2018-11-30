import json
import os
import random

import cv2
import numpy as np
import progressbar
from imutils import paths

# initialize the labels
from sklearn.model_selection import train_test_split

NUM_CLASSES = 2
NUM_TEST_IMAGES = 50 * NUM_CLASSES
NUM_VAL_IMAGES = int(50 * NUM_CLASSES / 2)

MX_OUTPUT = os.path.expanduser("~/data")
TRAIN_MX_LIST = os.path.sep.join([MX_OUTPUT, "lists/train.lst"])
VAL_MX_LIST = os.path.sep.join([MX_OUTPUT, "lists/val.lst"])
TEST_MX_LIST = os.path.sep.join([MX_OUTPUT, "lists/test.lst"])

DATASET_MEAN = os.path.expanduser("~/data/output/vggnet/pollinator_mean.json")

labels = []

# grab the image paths and randomly shuffle them
image_paths = sorted(list(paths.list_images("/code/media/pollinator_photos")))
random.seed(42)
random.shuffle(image_paths)

# Fix for paths that include spaces
image_paths = [img_path.replace("\\", "") for img_path in image_paths]

# Loop over the input images
for img_path in image_paths:
    # Extract the class label from the image path and update the
    # labels list
    label = img_path.split(os.path.sep)[-2]
    # Labels are defined based on being in the Not_Pollinator folder
    # because some images are saved in more specific folders like
    # Pollinator/Anthophora/img_name.png
    label = 0 if label == "Not_Pollinator" else 1
    labels.append(label)

labels = np.array(labels)

# Partition the data into training and testing splits based on the
# constants defined at the top of the file
(train_x, test_x, train_y, test_y) = train_test_split(image_paths, labels, test_size=NUM_TEST_IMAGES, random_state=42)

# Further split our training images so we have some validation images
# left over that our final model won't have seen
(train_x, val_x, train_y, val_y) = train_test_split(train_x, train_y, test_size=NUM_VAL_IMAGES, random_state=42)

# Construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output
# list files
datasets = [
    ("train", train_x, train_y, TRAIN_MX_LIST),
    ("val", val_x, val_y, VAL_MX_LIST),
    ("test", test_x, test_y, TEST_MX_LIST)
]

# Initialize the list of red, green, and blue channel averages
(r, g, b) = ([], [], [])

# Loop over the dataset tuples
for (d_type, paths, labels, output_path) in datasets:
    # Open the output file for writing
    print("[*] Building {}...".format(output_path))
    f = open(output_path, "w")

    # Initialize the progress bar
    widgets = ["Building List: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),
                                   widgets=widgets).start()

    # Loop over each of the individual images + labels
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # Write the image index, label, and output path to file
        row = "\t".join([str(i), str(label), path])
        f.write("{}\n".format(row))

        # If we are building the training dataset, then compute
        # the mean of each channel in the image, then update
        # the respective lists
        if d_type == "train":
            image = cv2.imread(path)
            _b, _g, _r = cv2.mean(image)[:3]
            r.append(_r)
            g.append(_g)
            b.append(_b)

        pbar.update(i)

    # Close the output file
    pbar.finish()
    f.close()

# Construct a dictionary of averages, then serialize the means
# to a JSON file
print("[*] Serializing means...")
channel_means = {"R": np.mean(r), "G": np.mean(g), "B": np.mean(b)}
f = open(DATASET_MEAN, "w")
f.write(json.dumps(channel_means))
f.close()

# Let the user know things finished successfully
print("[*] Done!")
