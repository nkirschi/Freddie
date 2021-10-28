# relevant directories !relative to project root!
DATA_DIR     = "data"         # where metadata about data is located
LABEL_SUBDIR = "labels"       # where the label mapping is located
RAW_SUBDIR   = "raw"          # where the raw orbit data is located
TRAIN_SUBDIR = "train"        # where the final training data resides

TEMP_DIR     = "temp"         # where temporary content is saved
RUNS_DIR     = "runs"         # where the training run data resides
MODELS_DIR   = "models"       # where the trained models reside

# important file names
MESSENGER = lambda n: f"messenger-{n:04d}"
ORBIT_FILE = lambda n: MESSENGER(n) + ".csv"
LABEL_FILE = "messenger-0001_-4094_labelled.csv"
STATS_FILE = "statistics.csv"
DIST_FILE = "class_distribution.csv"
HEALTH_FILE = "orbit_health.csv"
HPARAMS_FILE = "hyperparams.yaml"
TPARAMS_FILE = "techparams.yaml"

# important column names
DATE_COL = "DATE"
ORBIT_COL = "ORBIT"
LABEL_COL = "LABEL"
STAT_COL = "STAT"

EVENT_COLS = list(range(1, 9))
FLUX_COLS = ["BX_MSO", "BY_MSO", "BZ_MSO"]

# class name mapping
CLASSES = {
    0: "interplanetary magnetic field (IMF)",
    1: "bow shock crossing (SK)",
    2: "magnetosheath",
    3: "magnetopause crossing (MP)",
    4: "magnetosphere"
}