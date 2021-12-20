"""
This file contains the central dataset class.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"

from utils import constants as c
import torch
import numpy as np
import pandas as pd
import utils.io as ioutils
import random

from torch.utils.data import Dataset, Subset
from tqdm import tqdm


class MessengerDataset(Dataset):
    """
    The MESSENGER dataset, sliced into fixed-length sliding time windows.
    """

    def __init__(self, data_root, *, split, features, window_size, future_size, use_orbits=1.0, normalize=True):
        """
        Loads the MESSENGER dataset from orbits_path and configures it according to the parameters.

        Parameters
        ----------
        data_root : Path
            The path to the folder containing the MESSENGER data.
        split: str
            The dataset partition to use. Must be one of "train", "eval" or "test".
        features : list of str
            The names of features to include in the dataset. Available features:
            X_MSO, Y_MSO, Z_MSO, BX_MSO, BY_MSO, BZ_MSO, DBX_MSO, DBY_MSO, DBZ_MSO, RHO_DIPOLE, PHI_DIPOLE,
            THETA_DIPOLE, BABS_DIPOLE, BX_DIPOLE, BY_DIPOLE, BZ_DIPOLE, RHO, RXY, X, Y, Z, VX, VY, VZ, VABS, D, COSALPHA
        window_size : int
            The length each window shall have.
        future_size : int
            The number of time steps to predict into the future.
        use_orbits : float or list of int, default 1.0
            Either a percentage in (0,1) or a list of IDs of orbits to load. If 1.0, all orbits are loaded.
        normalize : bool, default True
            Whether to normalize the features.
        """

        # initialize attributes
        self.data_root = data_root
        self.split = split
        self.features = features
        self.window_size = window_size
        self.future_size = future_size
        self.use_orbits = use_orbits
        self.normalize = normalize

        # load data and metadata
        self.stats = self._load_stats()
        self.class_freqs = self._load_class_freqs()
        self.orbits = self._load_data()

        # determine indices of the first window for each orbit
        self.orbit_borders = self._determine_orbit_borders()

    def _load_stats(self):
        return pd.read_csv(ioutils.resolve_path(self.data_root) / c.STATS_FILE,
                           usecols=[c.STAT_COL].extend(self.features), index_col=c.STAT_COL)

    def _load_class_freqs(self):
        return pd.read_csv(ioutils.resolve_path(self.data_root) / c.FREQS_FILE, index_col=0)

    def _load_data(self):
        subdir = self.data_root
        if self.split == "train":
            subdir /= c.TRAIN_SUBDIR
        elif self.split == "eval":
            subdir /= c.EVAL_SUBDIR
        elif self.split == "test":
            subdir /= c.TEST_SUBDIR
        else:
            raise ValueError("invalid 'split' argument")

        if isinstance(self.use_orbits, list):
            files = map(lambda n: subdir / c.ORBIT_FILE(n), self.use_orbits)
        else:
            files = sorted(subdir.glob("*.csv"))
            if self.use_orbits < 1:
                files = random.sample(files, int(self.use_orbits * len(files)))

        orbits = []
        for file in tqdm(files):
            df_orbit = pd.read_csv(file, usecols=[c.LABEL_COL].extend(self.features),
                                   parse_dates=True, memory_map=True)
            if self.normalize:
                data = df_orbit[self.features]
                mean = self.stats.loc["mean"]
                std = self.stats.loc["std"]
                df_orbit.loc[:, self.features] = (data - mean) / std
            orbits.append(df_orbit)
        return orbits

    def _determine_orbit_borders(self):
        orbit_sizes = np.array(list(map(len, self.orbits)))  # original orbit lengths
        skip_size = (self.window_size + self.future_size) - 1  # end-of-orbit skip due to windowing
        return np.append(0, np.cumsum(orbit_sizes - skip_size))

    def __len__(self):
        """
        Yields the size of the dataset.

        Returns
        -------
        int
            The total number of samples.
        """

        return self.orbit_borders[-1]

    def __getitem__(self, idx):
        """
        Retrieves the window with the given index and its corresponding label.

        Parameters
        ----------
        idx : int
            The index of the desired window.

        Returns
        -------
        sample : tensor of float
            The requested window as float tensor.
        label : tensor of int
            The corresponding step-wise labels between 0 and 4.

        Raises
        ------
        IndexError
            Iff idx is negative or larger than the dataset size.
        """

        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} is out of bounds.")

        orbit_idx = np.sum(idx >= self.orbit_borders) - 1  # orbit index of the requested window
        pos = idx - self.orbit_borders[orbit_idx]  # window index within the orbit

        orbit = self.orbits[orbit_idx]
        window = orbit.iloc[pos:(pos + self.window_size)][self.features]
        target = orbit.iloc[pos:(pos + self.window_size + self.future_size)][c.LABEL_COL]

        sample = torch.as_tensor(window.values.transpose(), dtype=torch.float)
        label = torch.as_tensor(target.values, dtype=torch.long)

        return sample, label

    def explode_orbits(self):
        orb_range = lambda orbit_idx: range(self.orbit_borders[orbit_idx], self.orbit_borders[orbit_idx + 1])
        return {orb.iloc[0][c.ORBIT_COL]: Subset(self, orb_range(i)) for i, orb in enumerate(self.orbits)}

    def get_orbits(self):
        """
        Returns the orbits loaded by this dataset instance.

        Returns
        -------
        A list of all orbits used by this instance.
        """
        return self.orbits

    def get_class_frequencies(self):
        """
        Returns the frequency of each class.

        Returns
        -------
        A tensor containing the class frequencies.
        """
        return torch.as_tensor(self.class_freqs.values, dtype=torch.float)
