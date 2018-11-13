import argparse
import errno
import sys

import cv2
import numpy as np
import os

CLASSES = ["Pollinator", "Not_Pollinator", "Unknown"]


def get_next_image_path(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            yield os.path.join(root, name), name


def draw_multi_line_text(frame, text):
    h, w, _ = frame.shape
    y0, dy = int(h / 3), 50
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def setup_text(frame):
    h, w, _ = frame.shape
    left = np.zeros((h, int(w / 2), 3), dtype="uint8")
    right = np.zeros((h, int(w / 2), 3), dtype="uint8")
    bottom = np.zeros((int(h / 3), int(w / 2) * 2 + w, 3), dtype="uint8")

    left_text = "Left Arrow\nto Classify as\n" + CLASSES[1]
    right_text = "Right Arrow\nto Classify as\n" + CLASSES[0]
    bottom_text = "Down Arrow to Classify as " + CLASSES[2] + "\nPress 'q' to QUIT"

    draw_multi_line_text(left, left_text)
    draw_multi_line_text(right, right_text)
    draw_multi_line_text(bottom, bottom_text)

    return left, right, bottom


def main(**kwargs):
    # Create classification directories if they don't yet exist
    create_classification_folders(CLASSES, kwargs["write_path"])
    for image, name in get_image(kwargs["image_path"]):
        classify(kwargs["write_path"], name, image, image_path=kwargs["image_path"])


def classify(write_path, name, crop, image_path=None, all_negative=None):
    border_top = int(300 - crop.shape[0])
    border_bottom = border_top
    border_left = int(300 - crop.shape[1])
    border_right = border_left
    image = cv2.copyMakeBorder(crop, border_top, border_bottom, border_left, border_right, cv2.BORDER_CONSTANT,
                               None, 0)
    left, right, bottom = setup_text(image)
    window = np.vstack([np.hstack([left, image, right]), bottom])

    cv2.imshow("Possible Pollinator Image", window)

    while True:
        key = cv2.waitKey(0) & 0xFF or all_negative

        # if the left arrow key was pressed, move the image to the non-pollinator folder
        if key == 81:
            print("[*] Classifying {} as {}".format(name, CLASSES[1]))
            if image_path:
                os.rename(image_path, os.path.join(write_path, CLASSES[1], name))
            else:
                cv2.imwrite(os.path.join(write_path, CLASSES[1], name), crop)

            cv2.destroyWindow("Possible Pollinator Image")
            return CLASSES[1]

        # if the right arrow key was pressed, move the image to the pollinator folder
        elif key == 83:
            print("[*] Classifying {} as {}".format(name, CLASSES[0]))
            if image_path:
                os.rename(image_path, os.path.join(write_path, CLASSES[0], name))
            else:
                cv2.imwrite(os.path.join(write_path, CLASSES[0], name), crop)

            cv2.destroyWindow("Possible Pollinator Image")
            return CLASSES[0]

        # if the down arrow key was pressed, move the image to the unknown folder
        elif key == 84:
            print("[*] Classifying {} as {}".format(name, CLASSES[2]))
            if image_path:
                os.rename(image_path, os.path.join(write_path, CLASSES[2], name))
            else:
                cv2.imwrite(os.path.join(write_path, CLASSES[2], name), crop)

            cv2.destroyWindow("Possible Pollinator Image")
            return CLASSES[2]

        # if the `q` key was pressed, break from the loop
        elif key == ord("n"):
            print("[*] Classifying {} as {}".format(name, CLASSES[1]))
            if image_path:
                os.rename(image_path, os.path.join(write_path, CLASSES[1], name))
            else:
                cv2.imwrite(os.path.join(write_path, CLASSES[1], name), crop)

            cv2.destroyWindow("Possible Pollinator Image")
            return "No pollinators in frame."

        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            print("[!] Quitting!")
            sys.exit()

        else:
            print("[!] Invalid keypress.")


def get_image(path):
    for file_path, name in get_next_image_path(path):
        image = cv2.imread(file_path)

        yield image, name


def create_classification_folders(class_names, write_path):
    for classification in class_names:
        try:
            os.mkdir(os.path.join(write_path, classification))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image-path", type=str, required=True,
                    help="path to directory containing unclassified pollinator image files")
    ap.add_argument("-w", "--write-path", type=str, required=True,
                    help="path to write classified pollinator images")
    args = vars(ap.parse_args())
    main(**args)
