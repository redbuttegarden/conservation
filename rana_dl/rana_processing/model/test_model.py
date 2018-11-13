from keras.preprocessing.image import img_to_array
from keras.models import load_model
import multiprocessing
import numpy as np
import argparse
import imutils
import cv2


def get_model(model_path):
    # Load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(model_path)

    return model


def handle_prediction(return_dict, model_path, images):
    """
    Worker function.
    :param return_dict: Allows retrieval of vales from the process.
    :param model_path: File path where pre-trained model is saved.
    :param images: List of arrays representing images to be classified by model.
    :return: The values in the dictionary.
    """
    model = get_model(model_path)
    results = model.predict(images, verbose=1)
    return_dict[0] = results


def model_classifier(model_path, images):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=handle_prediction, args=(return_dict, model_path, images))
    p.start()
    p.join()
    return return_dict[0]


def pre_process(image):
    # Pre-process the image for classification
    image = cv2.resize(image, (84, 84))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to trained model")
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    args = vars(ap.parse_args())

    # Load the image
    image = cv2.imread(args["image"])
    orig = image.copy()

    image = pre_process(image)

    path = args["model"]
    model = get_model(path)

    # Classify the input image
    (not_pollinator, pollinator) = model_classifier(model, image)

    # Build the label
    label = get_label(not_pollinator, pollinator)

    # Draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # Show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)


def get_label(not_pollinator, pollinator):
    label = "Pollinator" if pollinator > not_pollinator else "Not Pollinator"
    return label


if __name__ == "__main__":
    main()
