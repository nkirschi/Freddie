import torch
import pandas as pd

from constants import *
from torch.utils.data import Dataset

# TODO write documentation
class MessengerDataset(Dataset):

    # TODO add parameters for 3D and 1D feature column names
    def __init__(self, orbits_path, window_size=8, normalize=True):
        self.data = pd.read_csv(orbits_path)
        self.window_size = window_size
        self.orbit_ids = self.data[ORBIT_COL].unique()
        self.max_num_windows = self.data.groupby(ORBIT_COL).size().min() - (window_size - 1)

        if normalize:  # TODO maybe extract special normalization into utils file
            for feat_3d in COLS_3D:
                centered = self.data[feat_3d] - self.data[feat_3d].mean()
                distdev = (centered ** 2).sum(axis=1).mean() ** 0.5
                self.data[feat_3d] = centered / distdev
            single = self.data[COLS_SINGLE]
            self.data[COLS_SINGLE] = (single - single.mean()) / single.std()

    def __len__(self):
        return len(self.orbit_ids) * self.max_num_windows

    def __getitem__(self, idx):
        o_idx, w_idx = divmod(idx, self.max_num_windows)

        orbit = self.data[self.data[ORBIT_COL] == self.orbit_ids[o_idx]]
        window = orbit.iloc[w_idx: (w_idx + self.window_size)]

        feats_3d = [torch.tensor(window[feat].values) for feat in COLS_3D]
        sample = torch.stack(feats_3d, dim=1)  # window_size x #feats_3d x 3
        label = torch.tensor(list(window[LABEL_COL]))  # list of ordinal labels

        return sample, label
