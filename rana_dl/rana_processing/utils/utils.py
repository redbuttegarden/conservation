import os
import random
import sys
from collections import namedtuple
from datetime import datetime
from os import walk

import cv2
import imutils
from imutils.contours import sort_contours
import numpy as np

Video = namedtuple('Video', ['directory', 'files'])


def check_frame_time(frame, frame_time, reference_digits, time_parsable, ts_box):
    if frame_time is not None:
        # If we succeeded in parsing the timestamp info in the frame, we set time_parsable to true
        # so we don't need to select the timestamp area in subsequent frames (until the next video)
        time_parsable = True
    else:
        # Try again
        frame_time = compute_frame_time(frame, reference_digits, time_parsable, ts_box)
    return frame_time, time_parsable


def classify_digits(img, reference_digits):
    img = imutils.resize(img, height=150)
    img_thresh = get_thresh(img)
    img_cnts, bboxes = get_contours(img_thresh, upper_thresh=11000)

    cv2.drawContours(img, img_cnts, -1, (0, 255, 0), 2)

    output = []
    for c, box in zip(img_cnts, bboxes):
        (x, y, w, h) = box
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))

        # Initialize a list of template matching scores
        scores = []

        # Loop over the reference digit name and digit ROI
        for (digit, digitROI) in reference_digits.items():
            # Apply correlation-based template matching, take the
            # score, and update the scores list
            result = cv2.matchTemplate(roi, digitROI,
                                       cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # The classification for the digit ROI will be the reference
        # digit name with the largest template matching score
        max_score = str(np.argmax(scores))
        output.append((max_score, roi))

    return output


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


def define_reference_digits(ref, ref_cnts, bounding_boxes):
    digits = {}
    # Loop over the OCR reference contours
    for (i, c) in enumerate(ref_cnts):
        # get the bounding box for the digit and resize it to a fixed size
        (x, y, w, h) = bounding_boxes[i]
        roi = ref[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))

        # update the digits dictionary, mapping the digit name to the ROI
        digits[i] = roi

    return digits


def get_contours(thresh, lower_thresh=2000, upper_thresh=5000):
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    sorted_cnts, bounding_boxes = sort_contours(cnts)
    final_contours = []
    bboxes = []
    # Filter contours
    for cnt, bbox in zip(sorted_cnts, bounding_boxes):
        area = cv2.contourArea(cnt)
        if upper_thresh > area > lower_thresh:
            final_contours.append(cnt)
            bboxes.append(bbox)

    return final_contours, bboxes


def get_filtered_frame(frame, background_extractor, blur_intensity=3):
    # Remove the text boxes
    frame = text_areas_removed(frame.copy())

    # Convert the filtered frame to gray for faster / more simple processing
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply some blur to lessen the affects of camera sensor noise
    frame = cv2.GaussianBlur(frame, (blur_intensity, blur_intensity), 0)

    # Remove the background
    frame = background_extractor.apply(frame)

    return frame


def get_frame_time(frame, reference_digits, timestamp_box):
    timestamp_area = get_timestamp_area(frame, timestamp_box)
    frame_time = process_timestamp_area(reference_digits, timestamp_area)
    return frame_time


def get_timestamp_area(frame, ts_box):
    """
    Get the cropped area surrounding the timestamp of an image.
    :param frame: A Numpy array representing the original image.
    :param ts_box: A tuple returned from OpenCV's selectROI function defining the coordinates around the timestamp of
    the image.
    :return: A Numpy array representing the cropped area of the image containing the timestamp information.
    """
    timestamp_area = frame[int(ts_box[1]):int(ts_box[1] + ts_box[3]), int(ts_box[0]):int(ts_box[0] + ts_box[2])]
    return timestamp_area


def get_timestamp_box(frame):
    print("[!] Please select the area around the video timestamp.")
    ts_box = cv2.selectROI("Timestamp Area Selection", frame, fromCenter=False,
                           showCrosshair=True)
    cv2.destroyAllWindows()
    return ts_box


def get_thresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    final = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return final


def process_reference_digits():
    ref = cv2.imread(os.path.join(os.path.dirname(__file__), "ref_digits.png"))
    # Take a threshold of the image before finding contours
    thresh = get_thresh(ref)
    # Get contours of the reference image. Each should represent a digit
    # for matching against each frame in the video stream
    reference_contours, ref_boxes = get_contours(thresh, upper_thresh=9000)
    # Get the reference digits
    ref_digits = define_reference_digits(ref, reference_contours, ref_boxes)

    return ref_digits


def process_timestamp_area(reference_digits, timestamp_area):
    (h, w) = timestamp_area.shape[:2]

    first_line = timestamp_area[:int(h / 2), :w]
    fl_classification = classify_digits(first_line, reference_digits)
    fl_labels = [digit[0] for digit in fl_classification]

    second_line = timestamp_area[int(h / 2):, :w]
    sl_classification = classify_digits(second_line, reference_digits)
    sl_labels = [digit[0] for digit in sl_classification]

    labels = ''.join(fl_labels + sl_labels)
    try:
        timestamp = datetime.strptime(labels[:-2], "%Y%m%d%H%M%S")
        print("[*] Processed time:", timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        return timestamp
    except ValueError:
        print("[!] Could not process time. Please try again.")
        timestamp = None
        return timestamp


def text_areas_removed(frame):
    frame[11:11 + 29, 138:138 + 403] = (0, 0, 0)
    frame[531:531 + 15, 138:138 + 161] = (0, 0, 0)
    frame[516:516 + 29, 760:760 + 130] = (0, 0, 0)

    return frame


def get_video_list(video_path):
    """
    Returns a randomly shuffled list of filenames in the given
    directory path. Note that despite the function name, there is
    currently no validation of file types to determine if the
    returned files are actually videos.
    :param video_path: Path to directory to walk for file paths.
    :return: A randomly shuffled list of file paths.
    """
    videos = []
    for (dirpath, dirnames, filenames) in walk(video_path):
        videos.append(Video(dirpath, filenames))

    random.shuffle(videos)
    return videos


def manual_selection(frame, frame_number):
    """
    Allows for manual selection of a pollinator in a given frame. The
    user is presented with a cv2 window displaying the frame in
    question and presented with several options for how to label it.

    The `p` key can be pressed to indicate that a pollinator is present
    in the frame and opens a new window with the same frame, where the
    user can draw a box around the pollinator. In this scenario, the
    function returns both the numpy array representing the cropped
    image drawn by the user along with the formatted bounding box
    coordinates of the crop in the form of "X Y W H".

    The `n` key can be pressed to indicate that no pollinators are
    present in the frame. In this case, pollinator is returned False
    and box is returned None.

    Pressing any other key passes and returns nothing.
    :param frame: The numpy array image of the entire frame.
    :param frame_number: The frame number in the video. Used only to
    help orient the user as to where they are in the video stream.
    :return: When the frame has been marked as containing a pollinator,
    returns a numpy array image of the selected pollinator and the
    associated bounding box information as a formatted string. When
    the frame has been marked as not containing a pollinator,
    pollinator is returned as False and bounding box info as None.
    """
    print("""
[*] Frame number {}. 
    If a pollinator is present, hit `p` to select its location. 
    If the frame DOES NOT have a pollinator, hit `n`.
    Press `q` to exit program.
    Otherwise, press any other key to continue.
        """
          .format(frame_number))
    cv2.imshow("Pollinator Check", frame)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('p'):
        pollinator_boxes = select_pollinator(frame)
        for pollinator_box in pollinator_boxes:
            x, y, w, h = pollinator_box
            box = get_formatted_box(x, y, w, h)
            pollinator = get_pollinator_area(frame, pollinator_box)

            yield pollinator, box

    elif key == ord('n'):
        pollinator = False
        box = None
        yield pollinator, box

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        print("[!] Quitting!")
        sys.exit()
    else:
        pass


def get_filename(frame_number, count, video, frame=False):
    if frame:
        file_name = "-".join([video[:-4], "frame", str(frame_number), str(count)]) + ".png"
    else:
        file_name = "-".join([video[:-4], str(frame_number), str(count)]) + ".png"
    return file_name


def select_pollinator(frame, multiple=False):
    cv2.destroyAllWindows()
    print("[!] Please select the area around the pollinator.")
    if multiple:
        pollinators = cv2.selectROIs("Pollinator Area Selection", frame, fromCenter=False,
                                     showCrosshair=True)
    else:
        pollinator = cv2.selectROI("Pollinator Area Selection", frame, fromCenter=False,
                                   showCrosshair=True)
        # Convert tuple to ndarray so the same type is returned regardless
        # of what the multiple parameter is set to.
        pollinators = np.array([pollinator], np.int32)

    return pollinators


def get_formatted_box(x, y, w, h):
    box = "{} {} {} {}".format(x, y, w, h)
    return box


def get_pollinator_area(frame, pollinator_box):
    pollinator_area = frame[int(pollinator_box[1]):int(pollinator_box[1] + pollinator_box[3]),
                            int(pollinator_box[0]):int(pollinator_box[0] + pollinator_box[2])]
    return pollinator_area
