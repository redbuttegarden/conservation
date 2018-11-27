import argparse
import time

import cv2
import numpy as np
import os

import imutils
from imutils.video import FileVideoStream

from log import setup, add_entry, get_last_entry
from pollinator_classifier import CLASSES, classify, create_classification_folders
from model.test_model import get_label, model_classifier, pre_process
from utils import get_contours, get_filename, get_formatted_box, get_frame_time, get_timestamp_box, manual_selection, \
    process_reference_digits, get_video_list

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--manual-selection", action="store_true",
                help="flag to enable manual pollinator selection")
ap.add_argument("-u", "--use-model", action="store_true",
                help="flag to enable pollinator classification using model")
ap.add_argument("-v", "--video-path", type=str, required=True,
                help="path to directory containing video files")
ap.add_argument("-w", "--write-path", type=str, required=True,
                help="path to write possible pollinator images")
args = vars(ap.parse_args())

FILE_LOG = "pollinator_video_log.json"

if args["use_model"]:
    model_path = "model/pollinator.model"


def main():
    # Setup for database logging
    setup()
    manual = args["manual_selection"]
    machine_learning = args["use_model"]
    create_classification_folders(CLASSES, args["write_path"])
    kernel = np.ones((5, 5), np.uint8)

    # The reference digits are computed based on a supplied reference photo
    # We assume the reference photo contains all the digits 0-9 from left to right
    reference_digits = process_reference_digits()

    for vdir in get_video_list(args["video_path"]):
        for video in vdir.files:
            last_log = check_logs(manual, video)
            if last_log:
                count = last_log.id
            else:
                # High initial count just to reduce the chances of accidentally overwriting existing files
                count = 30000
            print("[*] Analyzing video", video)
            vs = FileVideoStream(os.path.join(vdir.directory, video)).start()

            # The current frame number
            f_num = 0

            # Allow the buffer some time to fill
            time.sleep(1.0)

            # Motion tracking needs only be initialized once per video
            motion_saliency = None

            # Threshold values to control the sensitivity of contours found in the motion map
            # We assume that contours with extreme areas (very large or small) are not likely to be pollinators
            thresh1 = 100
            thresh2 = 10000

            # Is the timestamp in this video parsable yet?
            time_parsable = False
            ts_box = None

            while vs.more():
                frame = vs.read()

                # If the frame is None, the video is done being processed and we can move to the next one
                if frame is None:
                    break
                else:
                    f_num += 1

                if manual:
                    # We check if there are log entries indicating the frame has already been analyzed
                    if last_log is not None:
                        if f_num < last_log.frame:
                            print("[*] Frame number {} has already been analyzed. Waiting for frame number {}..."
                                  .format(f_num, last_log.frame))
                            # Continue to the next frame if the logs indicate we have analyzed frames later than this
                            # one
                            time.sleep(0.1)
                            continue

                    for pollinator, box in manual_selection(frame, f_num):
                        if pollinator is not None:
                            w, h, _ = pollinator.shape
                            area = w * h
                            file_name = get_filename(count, video)
                            cv2.imwrite(os.path.join(args['write_path'], "Pollinator", file_name), pollinator)
                            frame_time, ts_box = compute_frame_time(frame, reference_digits, time_parsable, ts_box)
                            frame_time, time_parsable = check_frame_time(frame, frame_time, reference_digits,
                                                                         time_parsable, ts_box)
                            if frame_time is None:
                                print("[!] Failed to process time, probably because the frame is distorted. "
                                      "Skipping frame...")
                                continue

                            add_entry(directory=vdir.directory,
                                      video=video,
                                      time=frame_time,
                                      classification="Pollinator",
                                      size=area,
                                      bbox=box,
                                      frame_number=f_num,
                                      name=file_name,
                                      manual=manual)
                            count += 1
                else:
                    # This block is executed once per video
                    if motion_saliency is None:
                        # Motion saliency tracks the important parts of the video by finding differences between the
                        # current frame and previous frames
                        motion_saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
                        motion_saliency.setImagesize(frame.shape[1], frame.shape[0])
                        motion_saliency.init()

                    # Even though this frame might be processed already, we still want to have the motion saliency
                    # object analyze it so that when we do get a frame we haven't seen before, the motion saliency
                    # object is can see the differences in it based on past frames
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    (success, motion_map) = motion_saliency.computeSaliency(gray)

                    # We check if there are log entries indicating the frame has already been analyzed
                    if last_log is not None:
                        if f_num < last_log.frame:
                            print("[*] Frame number {} has already been analyzed. Waiting for frame number {}..."
                                  .format(f_num, last_log.frame))
                            # Continue to the next frame if the logs indicate we have analyzed frames later than this
                            # one
                            continue

                    frame_time, ts_box = compute_frame_time(frame, reference_digits, time_parsable, ts_box)
                    frame_time, time_parsable = check_frame_time(frame, frame_time, reference_digits, time_parsable,
                                                                 ts_box)

                    count = analyze_motion(motion_map, frame, f_num, frame_time, kernel, thresh1, thresh2, video, count,
                                           vdir, machine_learning)

            vs.stop()

            # Video done being processed. Move to finished folder
            os.rename(os.path.join(vdir.directory, video), os.path.join(args["video_path"], "Processed Videos", video))

    cv2.destroyAllWindows()


def check_frame_time(frame, frame_time, reference_digits, time_parsable, ts_box):
    if frame_time is not None:
        # If we succeeded in parsing the timestamp info in the frame, we set time_parsable to true
        # so we don't need to select the timestamp area in subsequent frames (until the next video)
        time_parsable = True
    else:
        # Try again
        frame_time = compute_frame_time(frame, reference_digits, time_parsable, ts_box)
    return frame_time, time_parsable


def compute_frame_time(frame, reference_digits, time_parsable, ts_box):
    # time_parsable is False until we can successfully parse the datetime in the frame
    if time_parsable is False:
        # We make the frame larger and cut it in half to make it easier for the user to select the
        # timestamp area
        larger = imutils.resize(frame[int(frame.shape[1] / 2):], width=1500)
        # The ts_box is a tuple representing the points around the timestamp area that the user
        # indicated
        ts_box = get_timestamp_box(larger)
        # We then attempt to parse the timestamp area in the frame based on the reference digits
        frame_time = get_frame_time(larger, reference_digits, ts_box)

    else:
        # We need to keep resizing the frame so that the timestamp crop will match the ts_box that the
        #  user supplied in the beginning of the video
        larger = imutils.resize(frame[int(frame.shape[1] / 2):], width=1500)
        frame_time = get_frame_time(larger, reference_digits, ts_box)
    return frame_time, ts_box


def analyze_motion(motion_map, frame, f_num, frame_time, kernel, thresh1, thresh2, video, count, vdir,
                   machine_learning):

    # Map that represents differences from the previous frames as white areas
    motion_map = (motion_map * 255).astype("uint8")
    # This morph reduces noise in the image while preserving more substantial white areas
    morph = cv2.morphologyEx(motion_map, cv2.MORPH_OPEN, kernel)

    # Flag if the frame doesn't contain any pollinators
    all_neg = False

    possible_pollinators = []

    # Draw contours around the white spots that meet our threshold criteria
    cnts, bounding_boxes = get_contours(morph, lower_thresh=thresh1, upper_thresh=thresh2)
    for cnt, box in zip(cnts, bounding_boxes):
        x, y, w, h = box

        # Expand the box a bit to ensure the pollinator is in the crop
        x -= 20
        y -= 20
        w += 20
        h += 20

        # Crop the pollinator with some buffer around the bounding box, limited by frame dimensions
        crop = frame[max(0, y):min(frame.shape[0], y + h),
                     max(0, x):min(frame.shape[1], x + w)]

        # Get the area of the contour box
        area = cv2.contourArea(cnt)

        # Save the location of the bounding box for later reference
        box = get_formatted_box(x, y, w, h)

        # Create a copy of the frame for each contour so we can show the user one contour at a time
        pollinators = frame.copy()

        # Draw a rectangle around the potential pollinator
        cv2.rectangle(pollinators, (x, y), (x + w, y + h), (0, 0, 255), 1)

        if machine_learning:
            # A dictionary to hold info about this possible pollinator
            possible_pol = {}

            # Create a copy of the possible pollinator for the model to consume
            model_image = crop.copy()
            model_image = pre_process(model_image)

            possible_pol["Image"] = model_image
            possible_pol["Area"] = area
            possible_pol["Box"] = box

            possible_pollinators.append(possible_pol)
        else:
            file_name = get_filename(count, video)

            # Show the user the frame with the potential pollinator highlighted with a rectangle
            cv2.imshow("Video Frame", pollinators)

            if all_neg is False:
                # This function displayed the cropped out potential pollinator and allows the user to classify it
                classification = classify(args['write_path'], file_name, crop)

                if classification == "No pollinators in frame.":
                    all_neg = True
                    classification = CLASSES[1]
                # Whatever the user decided is recorded in the log database
                add_entry(directory=vdir.directory,
                          video=video,
                          time=frame_time,
                          name=file_name,
                          classification=classification,
                          size=area,
                          bbox=box,
                          frame_number=f_num)
            else:
                classification = CLASSES[1]
                # All negative is true, so we know every contour is not a pollinator
                add_entry(directory=vdir.directory,
                          video=video,
                          time=frame_time,
                          name=file_name,
                          classification=classification,
                          size=area,
                          bbox=box,
                          frame_number=f_num)
            count += 1

    if machine_learning and len(possible_pollinators) > 0:
        images = [pol["Image"] for pol in possible_pollinators]
        images = np.vstack(images)
        # Classify the cropped images
        results = model_classifier(model_path, images)

        print(results)

        for pol, result in zip(possible_pollinators, results):
            not_pollinator, pollinator = result
            proba = pollinator if pollinator > not_pollinator else not_pollinator
            proba *= 100

            # Build the label
            label = get_label(not_pollinator, pollinator)

            x, y, w, h = pol["Box"].split()

            if label == "Pollinator":
                # Draw a rectangle around the potential pollinator
                cv2.rectangle(frame, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 255, 0), 1)

                add_entry(directory=vdir.directory,
                          video=video,
                          time=frame_time,
                          classification="Pollinator",
                          proba=proba,
                          size=pol["Area"],
                          bbox=pol["Box"],
                          frame_number=f_num)
                count += 1
            else:
                cv2.rectangle(frame, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 0, 255), 1)

        cv2.imshow("Pollinators", frame)
        cv2.waitKey(0)

    return count


def check_logs(manual, video):
    last_update = get_last_entry(manual, video)
    if last_update is not None:
        print("[*] Last entry from", last_update.timestamp)
        print("[*] Waiting for frame that has not been processed...")
    return last_update


if __name__ == "__main__":
    main()
