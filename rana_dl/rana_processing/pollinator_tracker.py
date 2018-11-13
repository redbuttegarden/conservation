import sys

from centroid_tracker import CentroidTracker
from imutils.video import FileVideoStream
import numpy as np
import argparse
import time
import cv2

from model.test_network import get_model, model_classifier


def analyze_frame(frame, crop, centroid_tracker, model, confidence):
    (H, W) = frame.shape[:2]

    detections = []
    rects = []

    # Loop over the detections
    for i, detection in enumerate(detections):
        proba = model_classifier(model, detection)[1]
        # Filter out weak detections by ensuring the predicted
        # probability is greater than n minimum threshold
        if proba > confidence:
            # Compute the (x, y)-coordinates of the bounding box
            # for the object, then update the bounding box
            # rectangles list
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))

            # Draw a bounding box surrounding the object so we can
            # visualize it
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)

    # Update our centroid tracker using the computed set of
    # bounding box rectangles
    objects = centroid_tracker.update(rects)

    # Loop over the tracked objects
    for (object_id, centroid) in objects.items():
        # Draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(object_id)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key was pressed, break from the loop
    if key == ord("q"):
        sys.exit()


def main():
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to model file")
    ap.add_argument("-c", "--confidence", type=float, default=0.95,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # Initialize our centroid tracker and frame dimensions
    ct = CentroidTracker()

    # Load our serialized model from disk
    print("[INFO] Loading model...")
    model = get_model(args["model"])

    vs = FileVideoStream("/media/ave/ae722e82-1476-4374-acd8-a8a18ef8ae7a/Rana_vids/Nova_1/121/03-2018.05.25_06.24.23-17.mpg").start()
    time.sleep(2.0)

    # Loop over the frames from the video stream
    while vs.more():
        # Read the next frame from the video stream
        frame = vs.read()

        analyze_frame(frame, ct, model, args["confidence"])

    # Cleanup
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    main()
