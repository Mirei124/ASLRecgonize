import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

DATA_PATH = "./data/"
INPUT_PATH = "./data/"
LANDMARK_FILES_DIR = os.path.join(INPUT_PATH, "train_landmark_files")
TRAIN_FILE = os.path.join(INPUT_PATH, "train.csv")
JSON_FILE = os.path.join(INPUT_PATH, "sign_to_prediction_index_map.json")

SEQ_LEN = 96
ROWS_PER_FRAME = 543


def apply_autoflip(xy):
    left_start_index = 468
    right_start_index = 522
    length = xy[:, 0, 0].shape[0]
    l_num = length - torch.isnan(xy[:, left_start_index, 0]).sum()
    r_num = length - torch.isnan(xy[:, right_start_index, 0]).sum()

    # fmt: off
    if r_num < l_num:
        xy = torch.stack((-xy[:, :, 0], xy[:, :, 1]), dim=2)

        hand_indexes = torch.tensor((468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488))

        lipsUpperOuter_indexes = torch.tensor((291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61))
        lipsLowerOuter_indexes = torch.tensor((375, 321, 405, 314, 17, 84, 181, 91, 46))
        lipsUpperInner_indexes = torch.tensor((308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78))
        lipsLowerInner_indexes = torch.tensor((324, 318, 402, 317, 14, 87, 178, 88, 95))

    else:
        hand_indexes = torch.tensor((522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542))

        lipsUpperOuter_indexes = torch.tensor((61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291))
        lipsLowerOuter_indexes = torch.tensor((46, 91, 181, 84, 17, 314, 405, 321, 375))
        lipsUpperInner_indexes = torch.tensor((78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308))
        lipsLowerInner_indexes = torch.tensor((95, 88, 178, 87, 14, 317, 402, 318, 324))
    # fmt: on

    hand = xy[:, hand_indexes]

    lipsUpperOuter = xy[:, lipsUpperOuter_indexes]
    lipsLowerOuter = xy[:, lipsLowerOuter_indexes]
    lipsUpperInner = xy[:, lipsUpperInner_indexes]
    lipsLowerInner = xy[:, lipsLowerInner_indexes]

    x = torch.concat((hand, lipsUpperOuter, lipsLowerOuter, lipsUpperInner, lipsLowerInner), dim=1)
    # print("feature", x.shape)
    return x


def load_relevant_data_subset(pq_path):
    data_columns = ["x", "y", "z"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    data = data.fillna(0.0)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def normalize_by_midwayBetweenEyes(x):
    midwayBetweenEyes = x[:, 168]
    mask = ~torch.isnan(midwayBetweenEyes[:, 0])
    masked = midwayBetweenEyes[mask]  # .mean(keepdims=True)
    m = torch.mean(masked, dim=0)
    if torch.any(torch.isnan(m)):
        return x
    else:
        m = torch.unsqueeze(m, 0)
        m = torch.unsqueeze(m, 0)
    return x - m


class ASLDatasetRaw(Dataset):
    def __init__(self, is_train=True, max_seq_len=96) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        df = pd.read_csv(TRAIN_FILE)
        if is_train:
            df = df.iloc[: int(len(df) * 0.8), :]
        else:
            df = df.iloc[int(len(df) * 0.8) :, :]
        label_map = json.load(open(JSON_FILE, "r"))
        df["label"] = df["sign"].map(label_map)
        self.df = df
        self.length = len(df)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """读取并返回一个文件的数据.

        Args:
            idx (int): index

        Returns:
            data: 数据特征 形状为 (122, max_seq_len)
            label: 标签 0-255
        """
        row = self.df.iloc[idx, :]
        data = load_relevant_data_subset(os.path.join(INPUT_PATH, row.path))
        # (seq_len, 543, 3)
        data = torch.tensor(data)

        # preprocess
        xy = data[:, :, :2]
        xy = normalize_by_midwayBetweenEyes(xy)
        xy = apply_autoflip(xy)
        #  (seq_len, 61, 2) -> (channels, max_seq_len)
        xy = xy.flatten(1, 2).permute((1, 0))
        xy = nn.ConstantPad1d((0, max(0, self.max_seq_len - xy.shape[-1])), 0)(xy)[:, :96]

        label = torch.tensor(row.label)
        return xy, label

    def __len__(self) -> int:
        return self.length


class ASLDataset(Dataset):
    def __init__(self, is_train=True, max_seq_len=96) -> None:
        super().__init__()
        load_prefix = "train_" if is_train else "test_"
        self.all_data = np.load(load_prefix + "x.npy")
        self.all_label = np.load(load_prefix + "y.npy")
        self.length = self.all_data.shape[0]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        data, label = (
            torch.from_numpy(self.all_data[idx]),
            torch.tensor(self.all_label[idx], dtype=torch.long),
        )
        return data, label

    def __len__(self) -> int:
        return self.length


if __name__ == "__main__":
    data = ASLDatasetRaw(True)[10]
    print(data[0].shape)
    print(data[1].shape)
    print(data[1])
    print()

    data = ASLDataset(True)[10]
    print(data[0].shape)
    print(data[1].shape)
    print(data[1])
