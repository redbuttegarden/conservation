from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
from imutils import paths
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-i", "--images", required=True,
                help="path to input images")
args = vars(ap.parse_args())

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

image_paths = sorted(list(paths.list_images(args["images"])))

correct = 0
false_positive = 0  # When Not_Pollinator classified as a Pollinator
false_negative = 0  # When a Pollinator is classified as a Not_Pollinator
total = len(image_paths)

for path in image_paths:
    # load the image
    print("Reading from", path)
    image = cv2.imread(path)

    # pre-process the image for classification
    image = cv2.resize(image, (84, 84))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    truth = path.split(os.path.sep)[-2]
    print("True label =", truth)

    # classify the input image
    (not_pollinator, pollinator) = model.predict(image)[0]

    # build the label
    label = "Pollinator" if pollinator > not_pollinator else "Not_Pollinator"

    # Track number of correct classifications
    if label == truth:
        # Correctly classified image
        correct += 1
    elif truth == "Pollinator":
        # False negative
        false_negative += 1
    else:
        # False positive
        false_positive += 1

    # Format the label
    proba = pollinator if pollinator > not_pollinator else not_pollinator
    label = "{}: {:.2f}%".format(label, proba * 100)

    print("Predicted label:", label)

print("\n[*] Results:\nCorrect Classifications:", correct)
print("Total Incorrect Classifications:", total - correct)
print("False Negatives: {}\tFalse Positives: {}".format(false_negative, false_positive))
print("Accuracy:", correct / total)
