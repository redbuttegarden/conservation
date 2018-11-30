import argparse
import os
import random
from collections import Counter

from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="root path to dataset")
args = vars(ap.parse_args())

labels = []

# grab the image paths and randomly shuffle them
image_paths = sorted(list(paths.list_images(args["dataset"])))
random.shuffle(image_paths)

# Loop over the input images
for path in image_paths:
    # Extract the class label from the image path and update the
    # labels list
    label = path.split(os.path.sep)[-2]
    label = 0 if label == "Not_Pollinator" else 1
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
paired = zip(labels, image_paths)
sorted_paired = sorted(paired, key=lambda x: x[0])  # Sort by the label
# Unzip a subset of the data based on the difference between labels
_, over_represented_paths = zip(*sorted_paired[:or_freq - ur_freq])

for path in over_represented_paths:
    split = path.split(os.path.sep)
    top_dir = "/".join(split[:-3])
    over_represented_label = split[-2]
    name = split[-1]
    dst = os.path.join(top_dir, "overbalanced_extras", over_represented_label, name)
    print("Moving {} to {}".format(path, dst))
    os.rename(path, dst)
