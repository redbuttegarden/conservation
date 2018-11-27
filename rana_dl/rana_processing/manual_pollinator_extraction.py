import argparse
import cv2
import os
import time

from imutils.video import FileVideoStream
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

from pollinator_classifier import CLASSES, create_classification_folders
from utils import get_video_list, manual_selection, get_filename

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video-path", type=str, required=True,
                help="path to directory containing video files")
ap.add_argument("-w", "--write-path", type=str, required=True,
                help="path to write possible pollinator images")
args = vars(ap.parse_args())

create_classification_folders(CLASSES, args["write_path"])

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

species_path = os.path.join(args["write_path"], "Pollinator")
for species in id_options:
    species_join_path = os.path.join(species_path, species)
    print("[*] Checking {} for folder of {}.".format(species_path, species))
    if not os.path.exists(species_join_path):
        os.mkdir(species_join_path)
        print("Folder for {} wasn't found. Added as {}.".format(species, species_join_path))

for vdir in get_video_list(args["video_path"]):
    for video in vdir.files:
        print("[*] Analyzing video", video)
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

            for pollinator, box in manual_selection(frame, f_num):
                if pollinator is not None:
                    species_id = prompt("Visitor ID >> ", completer=pollinator_class_completer)
                    file_name = get_filename("-".join([str(f_num), str(count)]), video)
                    cv2.imwrite(os.path.join(args['write_path'], "Pollinator", species_id, file_name), pollinator)
                    count += 1

        vs.stop()

        # Video done being processed. Move to finished folder
        os.rename(os.path.join(vdir.directory, video), os.path.join(args["video_path"], os.pardir,
                                                                    "Processed Videos", video))

cv2.destroyAllWindows()
