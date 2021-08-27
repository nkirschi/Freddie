import torch
import pandas as pd
import utils

from constants import *
from torch.utils.data import Dataset


class MessengerDataset(Dataset):
    """
    The MESSENGER dataset, cut into fixed-length time windows.
    """

    # TODO add parameters for 3D and 1D feature column names
    def __init__(self, orbits_path, *, window_size=8, normalize=True):
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
        """

        self.data = pd.read_csv(orbits_path)
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
        idx
            The index of the desired window.

        Returns
        -------
        sample : bool
            The requested window as float tensor.
        label : int
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

        feats_3d = [torch.tensor(window[feat].values) for feat in COLS_3D]
        sample = torch.stack(feats_3d, dim=1)  # window_size x #feats_3d x 3
        label = torch.tensor(window[LABEL_COL].values)  # list of ordinal labels

        return sample, label
