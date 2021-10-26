import torch
import pandas as pd
import utils
import os

from constants import *
from itertools import chain
from torch.utils.data import Dataset, Subset


class MessengerDataset(Dataset):
    """
    The MESSENGER dataset, cut into fixed-length sliding time windows.
    """

    def __init__(self, data_path, *, features, window_size, future_size, use_orbits=None, normalize=True):
        """
        Loads the MESSENGER dataset from orbits_path and configures it according to the parameters.

        Parameters
        ----------
        data_path : str
            The path to the folder containing the MESSENGER data.
        features : list of str
            The names of features to include in the dataset. Available features:
            X_MSO, Y_MSO, Z_MSO, BX_MSO, BY_MSO, BZ_MSO, DBX_MSO, DBY_MSO, DBZ_MSO, RHO_DIPOLE, PHI_DIPOLE,
            THETA_DIPOLE, BABS_DIPOLE, BX_DIPOLE, BY_DIPOLE, BZ_DIPOLE, RHO, RXY, X, Y, Z, VX, VY, VZ, VABS, D, COSALPHA
        window_size : int
            The length each window shall have.
        future_size : int
            The number of time steps to predict into the future.
        use_orbits : list of int, default None
            The IDs of orbits to load. If None, all orbits are loaded.
        normalize : bool, default True
            Whether to normalize the features.
        """

        # initialize attributes
        self.data_path = data_path
        self.features = features
        self.window_size = window_size
        self.future_size = future_size
        self.use_orbits = use_orbits
        self.normalize = normalize

        # load data and metadata
        self.stats = self.__load_stats()
        self.class_dist = self.__load_class_dist()
        self.orbits = self.__load_data()

        # set auxiliary variables
        self.skip_size = (window_size + future_size) - 1  # how much is missing in the end
        self.orbit_sizes = pd.Series(len(orbit) for orbit in self.orbits)  # the original lengths of all orbits
        self.cum_sizes = (self.orbit_sizes - self.skip_size).cumsum()  # the cumulative final orbit lengths

    def __load_stats(self):
        return pd.read_csv(
            utils.resolve_path(self.data_path, STATS_FILE),
            usecols=[STAT_COL].extend(self.features),
            index_col=STAT_COL
        )

    def __load_class_dist(self):
        return pd.read_csv(utils.resolve_path(self.data_path, DIST_FILE), index_col=0)

    def __load_data(self):
        orbits = []
        for file in self.__determine_orbits():
            df_orbit = pd.read_csv(os.path.join(self.data_path, TRAIN_SUBDIR, file),
                                   usecols=[DATE_COL, LABEL_COL].extend(self.features),
                                   index_col=DATE_COL,
                                   parse_dates=True,
                                   memory_map=True)
            if self.normalize:
                data = df_orbit[self.features]
                mean = self.stats.loc["mean"]
                std = self.stats.loc["std"]
                df_orbit.loc[:, self.features] = (data - mean) / std
            orbits.append(df_orbit)
        return orbits

    def __determine_orbits(self):
        if self.use_orbits is None:
            return sorted(os.listdir(os.path.join(self.data_path, TRAIN_SUBDIR)))
        else:
            return map(lambda n: ORBIT_FILE(n), self.use_orbits)

    def __len__(self):
        """
        Yields the size of the dataset.

        Returns
        -------
        int
            The total number of samples.
        """

        return self.cum_sizes.iloc[-1]

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

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds.")

        orbit = (idx >= self.cum_sizes).sum()                        # orbit index of the requested window
        pos = idx - pd.Series(0).append(self.cum_sizes).iloc[orbit]  # window index within the orbit

        window = self.orbits[orbit].iloc[pos:(pos + self.window_size)]
        target = self.orbits[orbit].iloc[pos:(pos + self.window_size + self.future_size)]

        sample = torch.tensor(window[self.features].values, dtype=torch.float).transpose(0, 1)
        label = torch.tensor(target[LABEL_COL].values, dtype=torch.long)

        return sample, label

    def split(self, holdout_ratio):
        """
        Partitons the data into disjoint training and evaluation sets on the orbit level.

        Parameters
        ----------
        holdout_ratio : float
            The percentage of orbits that the evaluation subset shall cover.

        Returns
        -------
        (Subset, Subset)
            Two disjoint subsets of the data.
        """

        # shuffle the orbits
        cwn = self.cum_sizes
        orbits = torch.randperm(len(cwn))

        # split on the orbit level
        split_idx = int(holdout_ratio * len(orbits))
        train_orbits = orbits[split_idx:].tolist()
        test_orbits = orbits[:split_idx].tolist()

        # split on the index level
        orbit_range = lambda o: range(cwn.iloc[o - 1] if o > 0 else 0, cwn.iloc[o])
        train_indices = list(chain.from_iterable(orbit_range(o) for o in train_orbits))
        test_indices = list(chain.from_iterable(orbit_range(o) for o in test_orbits))

        return Subset(self, train_indices), Subset(self, test_indices)

    def get_class_distribution(self):
        return torch.tensor(self.class_dist.values, dtype=torch.float32)
