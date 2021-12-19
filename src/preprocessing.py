"""
This is the data preprocessing script.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"

###############################################################################

import pandas as pd
import random

from functools import partial
from os import PathLike
from tqdm import tqdm
from utils import constants as c, io


def prepare_labels():
    """Loads the label descriptor file and preprocesses it for merging."""

    df_labels = pd.read_csv(io.resolve_path(c.DATA_DIR) / c.LABEL_FILE,
                            index_col=c.ORBIT_COL, parse_dates=c.EVENT_COLS)

    # disregard incompletely labeled orbits
    df_labels = df_labels.dropna()

    # add columns for predecessor and successor crossings
    df_labels["Prev out"] = df_labels["SK outer out"].shift()
    df_labels["Next in"] = df_labels["SK outer in"].shift(-1)

    return df_labels


def prepare_orbit(orbit_id, *, df_labels, df_validity):
    """
    Loads the specified orbit and collects validity statistics for merging.

    Parameters
    ----------
    orbit_id: int
        The identifier of the orbit to prepare.
    df_labels: pd.DataFrame
        A pandas data frame giving the labels for all orbits.
    df_validity: pd.DataFrame
        A pandas data frame for writing the validity statistics to.

    Returns
    -------
    pd.DataFrame
        The fully loaded and prepared orbit. None iff the orbit in question is invalid.
    """

    orbit_file = io.resolve_path(c.DATA_DIR) / c.RAW_SUBDIR / c.ORBIT_FILE(orbit_id)
    df_validity.loc[orbit_id, "does_not_exist"] = not orbit_file.exists()

    if not df_validity.loc[orbit_id, "does_not_exist"]:
        df_orbit = pd.read_csv(orbit_file, index_col=c.DATE_COL, parse_dates=True)

        # record the number of NaN values in the orbit
        df_validity.loc[orbit_id, "total_nan_count"] = (
            df_orbit.isnull().sum().sum()
        )

        # record the number of missing time steps in the orbit
        df_validity.loc[orbit_id, "missing_entry_count"] = (
                (df_orbit.index[-1] - df_orbit.index[0]).total_seconds() + 1 - len(df_orbit)
        )

        # record whether the orbit has crossings from or into another orbit
        df_validity.loc[orbit_id, "has_special_conditions"] = (
                df_labels.loc[orbit_id, "SK outer in"] <= df_orbit.index[0]
                or df_orbit.index[0] <= df_labels.loc[orbit_id, "Prev out"]
                or df_labels.loc[orbit_id, "Next in"] <= df_orbit.index[-1]
                or df_orbit.index[-1] <= df_labels.loc[orbit_id, "SK outer out"]
        )

        # record the maximum occurring flux density magnitude
        df_validity.loc[orbit_id, "flux_norm_maximum"] = (
            df_orbit[c.FLUX_COLS].pow(2).sum(axis=1).pow(0.5).max()
        )

        # hacky but compact: check if the first three criteria are 0
        if df_validity.loc[orbit_id].iloc[0:-1].sum() == 0:
            df_orbit[c.ORBIT_COL] = orbit_id  # add orbit id
            return df_orbit

    return None  # rule out invalid orbit


def assign_labels(df_total, df_labels):
    """
    Assigns labels to the training time series using boundary labels.

    Parameters
    ----------
    df_total: pd.DataFrame
        The data frame containing all valid orbits.
    df_labels: pd.DataFrame
        The data frame giving the labels for all orbits.
    """

    # interplanetary magnetic field
    df_total[c.LABEL_COL] = 0

    for row in tqdm(range(len(df_labels))):
        time = lambda col: df_labels.iloc[row, df_labels.columns.get_loc(col)]

        # bow shock
        df_total.loc[time("SK outer in"):time("SK inner in"), c.LABEL_COL] = 1
        df_total.loc[time("SK inner out"):time("SK outer out"), c.LABEL_COL] = 1

        # magnetosheath
        df_total.loc[time("SK inner in"):time("MP outer in"), c.LABEL_COL] = 2
        df_total.loc[time("MP outer out"):time("SK inner out"), c.LABEL_COL] = 2

        # magnetopause
        df_total.loc[time("MP outer in"):time("MP inner in"), c.LABEL_COL] = 3
        df_total.loc[time("MP inner out"):time("MP outer out"), c.LABEL_COL] = 3

        # magnetosphere
        df_total.loc[time("MP inner in"):time("MP inner out"), c.LABEL_COL] = 4


def save_orbits(orbits, subdir):
    """
    Saves the given data frame orbit-wise in the specified subdirectory.

    Parameters
    ----------
    orbits: list of (int, pd.DataFrame)
        A list of pairs of orbit id and corresponding orbit data frame.
    subdir: PathLike
        A path specifying the subdirectory of the data directory to save to.
    """

    path = io.resolve_path(c.DATA_DIR) / subdir
    io.ensure_directory(path)
    for n, orbit in tqdm(orbits):
        orbit.to_csv(path / c.ORBIT_FILE(n))

################################################################################


# Prepare label descriptor file
df_labels = prepare_labels()
print(f"# fully labeled orbits: {len(df_labels)}")

# Prepare orbit validity statistics
df_validity = pd.DataFrame(
    index=df_labels.index,
    columns=["does_not_exist",
             "total_nan_count",
             "missing_entry_count",
             "has_special_conditions",
             "flux_norm_maximum"]
)

# Combine all orbits into one big frame
print("Loading and preparing all raw orbits...")
prepare_func = partial(prepare_orbit, df_labels=df_labels, df_validity=df_validity)
df_total = pd.concat(map(prepare_func, tqdm(df_labels.index)))

# Remove extreme outliers by the three-sigma rule
df_max = df_validity["flux_norm_maximum"]
outlier_orbits = df_validity.index[df_max > df_max.mean() + 3 * df_max.std()]
df_total = df_total[~df_total[c.ORBIT_COL].isin(outlier_orbits)]
print("Outlier orbits:", list(outlier_orbits))

# Save orbit validity statistics
df_validity.to_csv(io.resolve_path(c.DATA_DIR) / c.VALIDITY_FILE)
print(f"# healthy orbits: {len(df_total[c.ORBIT_COL].unique())}")

# Assign labels to the instances of the training data
print("Assigning labels to all time steps...")
assign_labels(df_total, df_labels)

# Save class distribution information
df_freqs = df_total.groupby(c.LABEL_COL).size().to_frame(c.FREQ_COL)
df_freqs.to_csv(io.resolve_path(c.DATA_DIR) / c.FREQS_FILE)
print("Class distribution:", df_freqs, sep="\n")

# Partition data on orbit level into
# - 70% training set
# - 20% evaluation set
# - 10% test set
orbits = list(df_total.groupby(c.ORBIT_COL))
random.shuffle(orbits)
i = int(0.7 * len(orbits))
j = int(0.9 * len(orbits))
train_orbits = orbits[:i]
eval_orbits = orbits[i:j]
test_orbits = orbits[j:]
print("# training set orbits:", len(train_orbits))
print("# evaluation set orbits:", len(eval_orbits))
print("# test set orbits:", len(test_orbits))

# Save descriptive statistics about the training set
df_train = pd.concat(list(zip(*train_orbits))[1])
df_stats = df_train.describe().rename_axis(c.STAT_COL)
df_stats.to_csv(io.resolve_path(c.DATA_DIR) / c.STATS_FILE)
print("Training set statistics:", df_stats, sep="\n")

# Save the final data frames orbit-wise
save_orbits(train_orbits, c.TRAIN_SUBDIR)
save_orbits(eval_orbits, c.EVAL_SUBDIR)
save_orbits(test_orbits, c.TEST_SUBDIR)
