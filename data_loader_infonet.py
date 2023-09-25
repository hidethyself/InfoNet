import os
import itertools
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class CustomLoader(Dataset):
    def __init__(
            self,
            feat_path,
            nb_channel=21,
            feat_shape=(300, 1344),
            data_loc="./data",
            type_="train"
    ):
        self.feat_path = feat_path
        self.nb_channel = nb_channel
        self.feat_shape = feat_shape
        self.nb_mel_bins = self.feat_shape[-1] // self.nb_channel
        self.time_steps = self.feat_shape[0]
        self.data_loc = data_loc
        self.type_ = type_
        self.csv_path = os.path.join(self.data_loc, '{}.csv'.format(self.type_))
        df = pd.read_csv(self.csv_path)
        self.file_names = df.values.tolist()
        del df
        self.file_names = list(itertools.chain(*self.file_names))
        self.feat_path = os.path.join(self.data_loc, self.feat_path)
        self.actual_feat_path = os.path.join(self.data_loc, "normalized_features_full_rank")
        self.label_path = os.path.join(self.data_loc, "label")

    def reshape_feature(self, feat):
        rearranged_feat = []
        for i in range(0, feat.shape[1] - 1, self.nb_mel_bins):
            rearranged_feat.append(feat[:, i: i + self.nb_mel_bins])
        rearranged_feat = np.asarray(rearranged_feat)
        return rearranged_feat

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        file_name = '{}.npy'.format(self.file_names[item].split(".")[0])
        feat = np.load(os.path.join(self.feat_path, file_name))
        feat = self.reshape_feature(feat=feat)
        act_feat = np.load(os.path.join(self.actual_feat_path, file_name))
        act_feat = self.reshape_feature(feat=act_feat)
        label = np.load(os.path.join(self.label_path, file_name))
        return feat, act_feat, label, self.file_names[item]


def load_data(
        batch_size,
        feat_path,
        data_loc="./data/env_10_7.5_3.5_rt_0.3",
        type_="train"
):
    print(f'Loading data from: {data_loc}')
    shuffle = True if type_ == "train" else False
    dataset = CustomLoader(
        data_loc=data_loc,
        feat_path=feat_path,
        type_=type_
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return data_loader


def test():
    train_loader = load_data(batch_size=4)
    for feat, act_feat, label, _ in train_loader:
        print(feat.shape, act_feat.shape, label.shape)


if __name__ == '__main__':
    test()