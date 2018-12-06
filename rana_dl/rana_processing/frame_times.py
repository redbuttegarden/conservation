import argparse
import os
import time

from imutils.video import FileVideoStream

from rana_logs.log import add_frame, get_processed_videos, setup
from utils.utils import check_frame_time, compute_frame_time, get_video_list, \
    process_reference_digits


def main(arguments):
    # Setup database tables
    setup()

    # The reference digits are computed based on a supplied reference photo
    # We assume the reference photo contains all the digits 0-9 from left to right
    reference_digits = process_reference_digits()

    # Is the timestamp in this video parsable yet?
    time_parsable = False
    ts_box = None

    processed_videos = get_processed_videos()

    for vdir in get_video_list(arguments["video_path"]):
        for video in vdir.files:
            if video in processed_videos:
                break

            print("[*] Processing video", video)
            vs = FileVideoStream(os.path.join(vdir.directory, video)).start()

            # Reset frame number
            f_num = 0

            # Allow the buffer some time to fill
            time.sleep(1.0)

            while vs.more():
                frame = vs.read()

                # If the frame is None, the video is done being processed and we can move to the next one
                if frame is None:
                    break
                else:
                    f_num += 1

                frame_time, ts_box = compute_frame_time(frame, reference_digits, time_parsable, ts_box)
                frame_time, time_parsable = check_frame_time(frame, frame_time, reference_digits, time_parsable,
                                                             ts_box)
                if frame_time is None:
                    print("[!] Failed to process time, probably because the frame is distorted.")

                add_frame(directory=vdir,
                          video=video,
                          time=frame_time,
                          frame_number=f_num)

            vs.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video-path", type=str, required=True,
                    help="path to directory containing video files")
    args = vars(ap.parse_args())

    main(args)
