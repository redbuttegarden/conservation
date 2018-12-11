import argparse
import cv2
import os
import time

from imutils.video import FileVideoStream
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

from rana_logs.log import add_log_entry, get_last_entry
from pollinator_classifier import CLASSES, create_classification_folders
from utils.utils import get_video_list, manual_selection, get_filename


def main(arguments):
    pollinator_class_completer = pollinator_setup(arguments)

    for vdir in get_video_list(arguments["video_path"]):
        for video in vdir.files:
            print("[*] Analyzing video", video)
            last_log = get_last_entry(True, video)

            vs = FileVideoStream(os.path.join(vdir.directory, video)).start()

            # The current frame number and pollinator count
            f_num = 0
            count = 0

            # Allow the buffer some time to fill
            time.sleep(1.0)

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
                        time.sleep(0.1)
                        continue

                for pollinator, box in manual_selection(frame, f_num):
                    frame_fname = get_filename(f_num, count, video, frame=True)
                    if pollinator is not False and pollinator is not None:
                        # Save the whole frame as a pollinator
                        print("[*] Saving frame as an example of Pollinator.")
                        cv2.imwrite(os.path.join(arguments["write_path"], "Frames", "Pollinator", frame_fname), frame)

                        # And save the pollinator
                        pol_fname = get_filename(f_num, count, video)
                        count = handle_pollinator(arguments, pol_fname, vdir, count, f_num, pollinator, box,
                                                  pollinator_class_completer, video)
                    elif pollinator is False and box is None:
                        # Save the whole frame as an example of no pollinator
                        print("[*] Saving frame as an example of Not_Pollinator.")
                        img_path = os.path.join(arguments["write_path"], "Frames", "Not_Pollinator", frame_fname)
                        cv2.imwrite(img_path, frame)
                        w, h, _ = frame.shape
                        size = w * h
                        print("[*] Logging this frame as Not_Pollinator.")
                        add_log_entry(directory=vdir.directory,
                                      video=video,
                                      time=None,
                                      classification="Not_Pollinator",
                                      proba=None,
                                      genus=None,
                                      species=None,
                                      behavior=None,
                                      size=size,
                                      bbox="Whole",  # Entire frame
                                      size_class=None,
                                      frame_number=f_num,
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
    id_options = ["Ant",
                  "Anthophora",
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
                  "Osmia1",
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
