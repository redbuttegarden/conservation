import os

NUM_CLASSES = 2
BATCH_SIZE = 32
NUM_DEVICES = 1
NUM_TEST_IMAGES = 50 * NUM_CLASSES
NUM_VAL_IMAGES = int(50 * NUM_CLASSES / 2)

MX_OUTPUT = os.path.expanduser("~/data")
TRAIN_MX_LIST = os.path.sep.join([MX_OUTPUT, "lists/train.lst"])
VAL_MX_LIST = os.path.sep.join([MX_OUTPUT, "lists/val.lst"])
TEST_MX_LIST = os.path.sep.join([MX_OUTPUT, "lists/test.lst"])

TRAIN_MX_REC = os.path.sep.join([MX_OUTPUT, "rec/train.rec"])
VAL_MX_REC = os.path.sep.join([MX_OUTPUT, "rec/val.rec"])
TEST_MX_REC = os.path.sep.join([MX_OUTPUT, "rec/test.rec"])

DATASET_MEAN = os.path.expanduser("~/data/output/vggnet/pollinator_mean.json")

LOG_OUTPUT = os.path.expanduser("~/data/output/vggnet")