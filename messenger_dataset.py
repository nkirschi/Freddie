import torch
import pandas as pd
import utils

from constants import *
from itertools import chain
from torch.utils.data import Dataset, Subset


class MessengerDataset(Dataset):
    """
    The MESSENGER dataset, cut into fixed-length time windows.
    """

    # TODO add parameters for 3D and 1D feature column names
    def __init__(self, orbits_path, *, window_size=8, normalize=True, partial=False):
        """
        Loads the MESSENGER dataset from orbits_path and configures it according to the parameters.

        Parameters
        ----------
        orbits_path : str
            The path to the folder containing all MESSENGER orbit data.
        window_size : int, default 8
            The size each window shall have.
        normalize : bool, default True
            Whether to normalize the features.
        partial : bool, default False
            Whether to only use a small fraction of the data.
        """

        data = pd.read_csv(orbits_path,
                           index_col=DATE_COL,
                           parse_dates=True,
                           dtype=DTYPES,
                           usecols=USED_COLS,
                           memory_map=True,
                           nrows=1e7 if partial else None)

        if partial:  # remove probably incomplete last group
            data = data[data[ORBIT_COL] != data.iloc[-1][ORBIT_COL]]

        self.data = data
        self.window_size = window_size
        self.skip_size = window_size - 1

        orbit_sizes = self.data.groupby(ORBIT_COL).size()
        self.cum_window_nums = (orbit_sizes - self.skip_size).cumsum()

        if normalize:
            for feat_3d in COLS_3D:
                utils.normalize_coupled(self.data, feat_3d)
            utils.normalize_decoupled(self.data, COLS_SINGLE)

    def __len__(self):
        """
        Yields the size of the dataset.

        Returns
        -------
        int
            The total number of windows.
        """

        return self.cum_window_nums.iloc[-1]

    def __getitem__(self, idx):
        """
        Retrieves the window the given index and its corresponding label.

        Parameters
        ----------
        idx : int
            The index of the desired window.

        Returns
        -------
        sample : tensor of float
            The requested window as float tensor.
        label : list of int
            The corresponding label between 0 and 4.

        Raises
        ------
        IndexError
            If idx is negative or larger than the dataset size.
        """

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds.")

        orbit_skips = (self.cum_window_nums <= idx).sum()
        pos = idx + orbit_skips * self.skip_size
        window = self.data.iloc[pos:(pos + self.window_size)]

        feats_3d = [torch.tensor(window[feat].values, dtype=torch.float) for feat in COLS_3D]
        sample = torch.stack(feats_3d, dim=1)  # window_size x #feats_3d x 3
        label = torch.tensor(window[LABEL_COL].values, dtype=torch.long)  # list of ordinal labels

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
        cwn = self.cum_window_nums
        orbits = torch.randperm(len(cwn))

        # split on the orbit level
        split_idx = int(holdout_ratio * len(orbits))
        train_orbits = orbits[:split_idx].tolist()
        test_orbits = orbits[split_idx:].tolist()

        # split on the index level
        orbit_range = lambda o: range(cwn.iloc[o - 1] if o > 0 else 0, cwn.iloc[o])
        train_indices = list(chain.from_iterable(orbit_range(o) for o in train_orbits))
        test_indices = list(chain.from_iterable(orbit_range(o) for o in test_orbits))

        return Subset(self, train_indices), Subset(self, test_indices)
