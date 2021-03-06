import os

NUM_CLASSES = 2
BATCH_SIZE = 4
NUM_TEST_IMAGES = 100 * NUM_CLASSES
NUM_VAL_IMAGES = int(100 * NUM_CLASSES / 2)

MX_OUTPUT = os.path.expanduser("/scratch/general/lustre/u6000791")
TRAIN_MX_LIST = os.path.sep.join([MX_OUTPUT, "lists/train.lst"])
VAL_MX_LIST = os.path.sep.join([MX_OUTPUT, "lists/val.lst"])
TEST_MX_LIST = os.path.sep.join([MX_OUTPUT, "lists/test.lst"])

TRAIN_MX_REC = os.path.sep.join([MX_OUTPUT, "rec/train.rec"])
VAL_MX_REC = os.path.sep.join([MX_OUTPUT, "rec/val.rec"])
TEST_MX_REC = os.path.sep.join([MX_OUTPUT, "rec/test.rec"])

DATASET_MEAN = "/scratch/general/lustre/u6000791/rec/pollinator_mean.json"

LOG_OUTPUT = os.path.expanduser("~/data/output")
