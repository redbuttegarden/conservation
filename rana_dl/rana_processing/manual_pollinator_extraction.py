import argparse
import cv2
import os
import time

import numpy as np
from imutils.video import FileVideoStream
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

from rana_logs.log import add_log_entry, get_last_entry
from pollinator_classifier import CLASSES, create_classification_folders
from utils.utils import get_video_list, manual_selection, get_filename


def handle_previous_frames(frame, previous_frames):
    """
    Maintains and returns a list of up to the previous 100 frames
    which also includes the most recent/current frame.
    :param frame: Current frame.
    :param previous_frames: The current list of previous frames.
    :return: A list of at most 100 previous frames including the most
    recent/current frame.
    """
    if len(previous_frames) >= 100:
        # Remove the oldest frame
        previous_frames.pop(0)

    # Add the current frame
    previous_frames.insert(0, frame)
    return previous_frames


def calculate_frame_number(labeled_frame, previous_frames, f_num):
    """
    Calculates the frame number the user labeled.
    :param labeled_frame: The frame labeled by the user.
    :param previous_frames: The list of at most the 100 previous
    frames including the most recent/current frame.
    :param f_num: The current frame number from the video stream.
    :return calc_fnum: An integer value indicating the frame number
    that was labeled by the user.
    """
    # Reverse the order of previous frames so recent frames are
    # located at the beginning of the list, allowing for list indexes
    # to be used to calculate the labeled frame number as an offset
    # of the current frame number.
    frame_idx = [np.array_equal(labeled_frame, frame) for frame in previous_frames].index(True)
    fnum_calc = f_num - frame_idx
    return fnum_calc


def main(arguments):
    pollinator_class_completer = pollinator_setup(arguments)

    for vdir in get_video_list(arguments["video_path"]):
        split = vdir.directory.split("/")[-2:]  # Extract site and plant info from directory path
        site = split[0]
        plant = split[1]
        for video in vdir.files:
            print("[*] Analyzing video {} from site {}, plant number {}.".format(video, site, plant))
            last_log = get_last_entry(True, video)

            vs = FileVideoStream(os.path.join(vdir.directory, video)).start()

            # The current frame number and pollinator count
            f_num = 0
            count = 0

            # Allow the buffer some time to fill
            time.sleep(2.0)

            # Keep a list of previous frames
            previous_frames = []

            while vs.more():
                frame = vs.read()

                # If the frame is None, the video is done being processed and we can move to the next one
                if frame is None:
                    break
                else:
                    f_num += 1
                    if last_log is not None and f_num <= last_log.frame:
                        print("[*] Frame number {} has already been analyzed. Waiting for frame number {}..."
                              .format(f_num, last_log.frame + 1))
                        # Continue to the next frame if the logs indicate we have analyzed frames later than this
                        # one
                        time.sleep(0.01)  # Sleep here so we don't overtake the buffer
                        continue

                """
                Because previous frames are passed to manual selection,
                the pollinator selection may not have occurred on the
                current frame. Therefore, the frame number used for
                file names and logging will need to be calculated.
                """
                previous_frames = handle_previous_frames(frame, previous_frames)
                pollinator, box, labeled_frame = manual_selection(f_num, previous_frames)
                if pollinator is None and box is None and labeled_frame is None:
                    continue

                fnum_calc = calculate_frame_number(labeled_frame, previous_frames, f_num)
                frame_fname = get_filename(fnum_calc, count, video, frame=True)
                if pollinator is not False and pollinator is not None:
                    # Save the whole frame as a pollinator
                    print("[*] Saving frame as an example of Pollinator.")
                    cv2.imwrite(os.path.join(arguments["write_path"], "Frames", "Pollinator", frame_fname),
                                labeled_frame)

                    # And save the pollinator
                    pol_fname = get_filename(fnum_calc, count, video)
                    count = handle_pollinator(arguments, pol_fname, vdir, count, fnum_calc, pollinator, box,
                                              pollinator_class_completer, video)
                elif pollinator is False and box is None:
                    # Save the whole frame as an example of no pollinator
                    print("[*] Saving frame as an example of Not_Pollinator.")
                    img_path = os.path.join(arguments["write_path"], "Frames", "Not_Pollinator", frame_fname)
                    cv2.imwrite(img_path, labeled_frame)
                    w, h, _ = frame.shape
                    size = w * h
                    print("[*] Logging this frame as Not_Pollinator.")
                    add_log_entry(directory=vdir.directory,
                                  video=video,
                                  time=None,
                                  classification="Not_Pollinator",
                                  pollinator_id=None,
                                  proba=None,
                                  genus=None,
                                  species=None,
                                  behavior=None,
                                  size=size,
                                  bbox="Whole",  # Entire frame
                                  size_class=None,
                                  frame_number=fnum_calc,
                                  manual=True,
                                  img_path=img_path,
                                  )

            vs.stop()

    cv2.destroyAllWindows()


def handle_pollinator(arguments, file_name, vdir, count, f_num, pollinator, box, pollinator_class_completer, video):
    w, h, _ = pollinator.shape
    area = w * h
    pol_id = prompt("Visitor ID >> ", completer=pollinator_class_completer)
    img_path = os.path.join(arguments["write_path"], "Pollinator", pol_id, file_name)
    print("[*] Saving pollinator image to", img_path)
    cv2.imwrite(img_path, pollinator)

    print("[*] Adding log entry to database...")
    add_log_entry(directory=vdir.directory,
                  video=video,
                  time=None,  # This will be populated later since timestamps are being preprocessed
                  name=file_name,
                  classification="Pollinator",
                  pollinator_id=pol_id,
                  proba=None,
                  genus=None,
                  species=None,
                  behavior=None,
                  size=area,
                  bbox=box,
                  size_class=None,
                  frame_number=f_num,
                  manual=True,
                  img_path=img_path,
                  )
    count += 1
    return count


def pollinator_setup(arguments):
    create_classification_folders(CLASSES, arguments["write_path"])
    id_options = ["Anthophora",
                  "Bee tiny",
                  "Bombylius",
                  "Butterfly",
                  "Fly",
                  "Halictus",
                  "Hyles lineata",
                  "Masarinae",
                  "Mosquito",
                  "Osmia",
                  "Osmia green",
                  "Unknown",
                  "Unknown bee",
                  "Unknown wasp",
                  "Wasp black",
                  "Xylocopa"]
    pollinator_class_completer = WordCompleter(id_options)
    species_path = os.path.join(arguments["write_path"], "Pollinator")
    for species in id_options:
        species_join_path = os.path.join(species_path, species)
        print("[*] Checking {} for folder of {}.".format(species_path, species))
        if not os.path.exists(species_join_path):
            os.mkdir(species_join_path)
            print("Folder for {} wasn't found. Added as {}.".format(species, species_join_path))
    return pollinator_class_completer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video-path", type=str, required=True,
                    help="path to directory containing video files")
    ap.add_argument("-w", "--write-path", type=str, required=True,
                    help="path to write possible pollinator images")
    args = vars(ap.parse_args())
    main(args)
