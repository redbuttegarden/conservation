import argparse
import os
import time

from imutils.video import FileVideoStream

from rana_logs.log import add_frame, get_last_processed_frame, setup, get_analyzed_videos, get_processed_videos, \
    add_processed_video
from utils.utils import check_frame_time, compute_frame_time, get_video_list, \
    process_reference_digits


def main(arguments):
    # Setup database tables
    setup()

    # The reference digits are computed based on a supplied reference photo
    # We assume the reference photo contains all the digits 0-9 from left to right
    reference_digits = process_reference_digits()

    # Is the timestamp in this video parsable yet?
    time_parsable = True
    ts_box = (1168, 246, 314, 73)

    analyzed_videos = get_analyzed_videos()
    processed_videos = get_processed_videos()

    for vdir in get_video_list(arguments["video_path"]):
        for video in vdir.files:
            if video in processed_videos:
                print("[*] Video has been fully processed. Skipping...")
                continue
            else:
                process_video(analyzed_videos, reference_digits, time_parsable, ts_box, vdir, video)


def process_video(analyzed_videos, reference_digits, time_parsable, ts_box, vdir, video):
    print("[*] Processing video {} from {}".format(video, vdir.directory))
    if video in analyzed_videos:
        print("[*] Video has been processed. Checking if processing is complete...")
        last_processed_frame = get_last_processed_frame(video)
        print("[*] Last processed frame for this video is: ", last_processed_frame)
    else:
        last_processed_frame = None

    vs = FileVideoStream(os.path.join(vdir.directory, video)).start()

    # Reset frame number
    f_num = 0

    # Allow the buffer some time to fill
    time.sleep(2.0)

    while vs.more():
        frame = vs.read()

        # If the frame is None, the video is done being processed and we can move to the next one
        if frame is None:
            print("{} is done being processed. Adding to database of processed videos...")
            add_processed_video(directory=vdir.directory,
                                video=video,
                                total_frames=f_num)
            break
        else:
            f_num += 1
            if (last_processed_frame is not None) and (f_num <= last_processed_frame):
                print("[*] Current frame is {}. Waiting for {}...".format(f_num, last_processed_frame + 1))
                # Give video buffer time to fill so we don't overtake it
                time.sleep(0.01)
                continue
            else:
                # Process the timestamp area in the video
                frame_time, ts_box = compute_frame_time(frame, reference_digits, time_parsable, ts_box)
                frame_time, time_parsable = check_frame_time(frame, frame_time, reference_digits, time_parsable,
                                                             ts_box)

                print(frame_time)  # TODO remove after debugging
                if frame_time is None:
                    print("[!] Failed to process time, probably because the frame is distorted.")

                # Add the frame information to the logging database
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