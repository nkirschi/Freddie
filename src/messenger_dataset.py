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

import constants as c
import torch
import numpy as np
import pandas as pd
import utils.ioutils as ioutils
import random

from itertools import chain
from torch.utils.data import Dataset, Subset


class MessengerDataset(Dataset):
    """
    The MESSENGER dataset, sliced into fixed-length sliding time windows.
    """

    def __init__(self, data_root, *, features, window_size, future_size, use_orbits=1.0, normalize=True, train=True):
        """
        Loads the MESSENGER dataset from orbits_path and configures it according to the parameters.

        Parameters
        ----------
        data_root : Path
            The path to the folder containing the MESSENGER data.
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
        train : bool, default True
            Whether to use the train or test data.
        """

        # initialize attributes
        self.data_root = data_root
        self.features = features
        self.window_size = window_size
        self.future_size = future_size
        self.use_orbits = use_orbits
        self.normalize = normalize
        self.train = train

        # load data and metadata
        self.stats = self._load_stats()
        self.class_dist = self._load_class_dist()
        self.orbits = self._load_data()

        # the cumulative number of windows per orbit
        self.orbit_borders = self._determine_orbit_borders()

    def _load_stats(self):
        return pd.read_csv(
            ioutils.resolve_path(self.data_root) / c.STATS_FILE,
            usecols=[c.STAT_COL].extend(self.features),
            index_col=c.STAT_COL
        )

    def _load_class_dist(self):
        return pd.read_csv(ioutils.resolve_path(self.data_root) / c.FREQS_FILE, index_col=0)

    def _load_data(self):
        orbits = {}
        subdir = c.TRAIN_SUBDIR if self.train else c.TEST_SUBDIR
        for file in self._determine_orbits():
            df_orbit = pd.read_csv(self.data_root / subdir / file,
                                   usecols=[c.DATE_COL, c.LABEL_COL].extend(self.features),
                                   parse_dates=True,
                                   memory_map=True)
            if self.normalize:
                data = df_orbit[self.features]
                mean = self.stats.loc["mean"]
                std = self.stats.loc["std"]
                df_orbit.loc[:, self.features] = (data - mean) / std
            orbits[df_orbit.loc[0, c.ORBIT_COL]] = df_orbit
        return orbits

    def _determine_orbits(self):
        if isinstance(self.use_orbits, list):
            return map(lambda n: c.ORBIT_FILE(n), self.use_orbits)
        else:
            orbits = sorted((self.data_root / c.TRAIN_SUBDIR).glob("*.csv"))
            if self.use_orbits < 1:
                orbits = random.sample(orbits, int(self.use_orbits * len(orbits)))
            return orbits

    def _determine_orbit_borders(self):
        # the original lengths of all orbits
        orbit_sizes = np.array([len(orbit) for orbit in self.orbits.values()])
        skip_size = (self.window_size + self.future_size) - 1
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
        label : list of int
            The corresponding step-wise labels between 0 and 4.

        Raises
        ------
        IndexError
            If idx is negative or larger than the dataset size.
        """

        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} is out of bounds.")

        orbit_idx = np.sum(idx >= self.orbit_borders) - 1  # orbit index of the requested window
        orbit = list(self.orbits.values())[orbit_idx]      # orbit data frame corresponding to the index
        pos = idx - self.orbit_borders[orbit_idx]          # window index within the orbit

        window = orbit.iloc[pos:(pos + self.window_size)][self.features]
        target = orbit.iloc[pos:(pos + self.window_size + self.future_size)][c.LABEL_COL]

        sample = torch.tensor(window.values.transpose(), dtype=torch.float)
        label = torch.tensor(target.values, dtype=torch.long)

        return sample, label

    def split(self, train_split):
        """
        Partitons the data into disjoint training and evaluation sets on the orbit level.

        Parameters
        ----------
        train_split : float or list of int
            Either the percentage of orbits or a concrete list of orbit IDs the training subset shall cover.

        Returns
        -------
        (Subset, Subset)
            Two disjoint subsets of the data.
        """

        if isinstance(train_split, list):
            keys = list(self.orbits.keys())
            train_orbits = [keys.index(o) for o in train_split]
            test_orbits = [keys.index(o) for o in list(set(self.orbits.keys()) - set(train_split))]
        else:
            # shuffle the orbits
            orbits = torch.randperm(len(self.orbits))

            # split on the orbit level
            split_idx = int(train_split * len(orbits))
            train_orbits = orbits[:split_idx].tolist()
            test_orbits = orbits[split_idx:].tolist()

        # split on the index level
        train_indices = list(chain.from_iterable(self._orbit_range(o) for o in train_orbits))
        test_indices = list(chain.from_iterable(self._orbit_range(o) for o in test_orbits))

        return Subset(self, train_indices), Subset(self, test_indices)

    def explode_orbits(self):
        return {key: Subset(self, self._orbit_range(i)) for i, key in enumerate(self.orbits)}

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
        return torch.tensor(self.class_dist.values, dtype=torch.float)

    def _orbit_range(self, orbit_idx):
        return range(self.orbit_borders[orbit_idx], self.orbit_borders[orbit_idx + 1])
