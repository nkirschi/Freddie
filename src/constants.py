"""
This file defines important project-wide constants.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"


# relevant directories !relative to project root!
CONFIG_DIR   = "config"  # where the configuration files are held
DATA_DIR     = "data"    # where metadata about data is located
LABEL_SUBDIR = "labels"  # where the label mapping is located
RAW_SUBDIR   = "raw"     # where the raw orbit data is located
TRAIN_SUBDIR = "train"   # where the final training data resides
TEST_SUBDIR  = "test"    # where the final testing data resides

RUNS_DIR     = "runs"         # where the training run data resides
CKPT_SUBDIR  = "checkpoints"  # where the trained models reside

TEMP_DIR     = "temp"         # where temporary content is saved

# important file names
MESSENGER = lambda n: f"messenger-{n:04d}"
ORBIT_FILE = lambda n: MESSENGER(n) + ".csv"
LABEL_FILE = "messenger-0001_-4094_labelled.csv"
STATS_FILE = "statistics.csv"
FREQS_FILE = "class_frequencies.csv"
HEALTH_FILE = "orbit_health.csv"
HPARAMS_FILE = "hyperparams.yaml"
TPARAMS_FILE = "techparams.yaml"
METRICS_FILE = "metrics.json"
BEST_MODEL_FILE = "best_model.pth"
CKPT_FILE = lambda n: f"epoch_{n:02d}.pth"
RUN_NAME = lambda n: f"{n:04d}"

# important column names
DATE_COL = "DATE"
ORBIT_COL = "ORBIT"
LABEL_COL = "LABEL"
STAT_COL = "STAT"

EVENT_COLS = list(range(1, 9))
FLUX_COLS = ["BX_MSO", "BY_MSO", "BZ_MSO"]

# class name mapping
CLASSES = {
    0: "IMF",
    1: "SK",
    2: "MSh",
    3: "MP",
    4: "MSp"
}