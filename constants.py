# relevant directories
LABEL_DIR = "labels"   # where the label mapping is located
ORBIT_DIR = "orbits"   # where the orbit data is located
MERGED_DIR = "merged"  # where the final training data resides
FIGURE_DIR = "figures"

# important file names
MESSENGER = lambda n: f"messenger-{n:04d}"
ORBIT_FILE = lambda n: MESSENGER(n) + ".csv"
LABEL_FILE = "messenger-0001_-4094_labelled.csv"
TRAIN_FILE = "df_train.csv"

# important column names
DATE_COL = "DATE"
ORBIT_COL = "ORBIT"
LABEL_COL = "LABEL"

# relevant feature column names
COLS_3D = [
    ["X", "Y", "Z"],
    ["VX", "VY", "VZ"],
    ["X_MSO", "Y_MSO", "Z_MSO"],
    ["BX_MSO", "BY_MSO", "BZ_MSO"],
    ["DBX_MSO", "DBY_MSO", "DBZ_MSO"],
    ["BX_DIPOLE", "BY_DIPOLE", "BZ_DIPOLE"],
]
COLS_SINGLE = [
    "RHO_DIPOLE",
    "PHI_DIPOLE",
    "THETA_DIPOLE",
    "BABS_DIPOLE",
    "RHO", "RXY",
    "VABS", "D",
    "COSALPHA"
]
FLUX_COLS = COLS_3D[3]

# class name mapping
LABEL_NAMES = {
    0: "interplanetary magnetic field",
    1: "bow shock crossing",
    2: "magnetosheath",
    3: "magnetopause crossing",
    4: "magnetosphere"
}

# event columns in label file
EVENT_COLS = list(range(1, 9))
