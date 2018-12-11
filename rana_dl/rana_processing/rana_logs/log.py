import os
from peewee import *

db = SqliteDatabase(os.path.join(os.path.dirname(__file__), 'database/log.db'))


class Frame(Model):
    id = PrimaryKeyField()
    directory = CharField()
    video = CharField()
    timestamp = DateTimeField(null=True)  # Null entries indicate frame time can't be processed
    frame = IntegerField()

    class Meta:
        database = db


class LogEntry(Model):
    id = PrimaryKeyField()
    directory = CharField()
    video = CharField()
    timestamp = DateTimeField(null=True)
    name = CharField(null=True)
    classification = CharField()
    probability = FloatField(null=True)
    genus = CharField(null=True)
    species = CharField(null=True)
    behavior = CharField(null=True)
    size = FloatField()
    bbox = CharField()
    size_class = CharField(null=True)
    frame = IntegerField()
    manual = BooleanField(default=False)
    img_path = CharField(null=True)

    class Meta:
        database = db


@db.connection_context()
def add_frame(directory, video, time, frame_number):
    frame_info = Frame(directory=directory,
                       video=video,
                       timestamp=time,
                       frame=frame_number)
    frame_info.save()


@db.connection_context()
def add_log_entry(directory, video, time, classification, size, bbox, frame_number, name=None, proba=None, genus=None,
                  species=None, behavior=None, size_class=None, manual=False, img_path=None):
    entry = LogEntry(directory=directory,
                     video=video,
                     timestamp=time,
                     name=name,
                     classification=classification,
                     probability=proba,
                     genus=genus,
                     species=species,
                     behavior=behavior,
                     size=size,
                     bbox=bbox,
                     size_class=size_class,
                     frame=frame_number,
                     manual=int(manual),
                     img_path=img_path
                     )
    entry.save()


@db.connection_context()
def get_last_entry(manual, video):
    """
    Returns the last entry in the database that was from the
    same video and inserted using the same method.
    :param manual: A boolean indicating if manual selection is being used.
    :param video: The name of the video file
    :return: Return the latest entry in the database that shares the same
    boolean value as the manual parameter.
    """
    try:
        last = LogEntry.select().where(LogEntry.video == video, LogEntry.manual == manual)\
            .order_by(LogEntry.id.desc()).get()

        return last
    except DoesNotExist:
        print("[*] No existing log entry for {} while manual selection is set to {}.".format(video, manual))


@db.connection_context()
def get_last_processed_frame(video):
    video_frames = Frame.select().where(Frame.video == video)
    f_nums = [vf.frame for vf in video_frames]
    last_frame = max(f_nums)
    return last_frame


@db.connection_context()
def get_processed_videos():
    """
    Returns a list of all videos that are referenced in the Frame table.
    Note that this list could contain videos that were only partially
    processed.
    :return: list of videos in Frame table.
    """
    try:
        videos = Frame.select()
        videos = set([vid.video for vid in videos])
        return videos
    except DoesNotExist:
        print("[*] No processed videos found.")


@db.connection_context()
def setup():
    db.create_tables([Frame, LogEntry])
