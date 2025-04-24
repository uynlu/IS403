import torch
from torch.utils.data import Dataset

import pandas as pd
import os
import joblib

from sklearn.preprocessing import MinMaxScaler


class CustomDataset(Dataset):
    def __init__(self,
        dataset_path: str,
        scaler_path: str,
        look_back: int = 30,
        n_steps: int = 1
    ):
        super(CustomDataset, self).__init__()
        self.dataset = pd.read_csv(dataset_path)
        self.look_back = look_back
        self.n_steps = n_steps
        self.scaler_path = scaler_path

        if (
            os.path.isfile(os.path.join(self.scaler_path, "features_scaler.pkl")) and 
            os.path.isfile(os.path.join(self.scaler_path, "scaler.pkl"))
        ):
            self.features_scaler = joblib.load(os.path.join(self.scaler_path, "features_scaler.pkl"))
            self.scaler = joblib.load(os.path.join(self.scaler_path, "scaler.pkl"))
            self.scale_transform()
        else:
            self.features_scaler = MinMaxScaler()
            self.scaler = MinMaxScaler()
            self.scale_fit_transform()

        self.create_dataset()

    def __len__(self):
        return len(self.features)

    def __getitem__(
        self,
        index: int
    ):
        return self.features[index], self.target[index]

    def create_dataset(self):
        features = torch.Tensor(self.dataset.values)
        targets = torch.Tensor(self.dataset["Close"])

        self.features, self.target = [], []
        for i in range(len(self.dataset) - self.look_back):
            X = features[i : i + self.look_back, :]
            y = targets[(i + self.look_back) : (i + self.look_back + self.n_steps)] 
            
            if len(y) == self.n_steps:
                self.features.append(X)
                self.target.append(y)

        self.features = torch.stack(self.features)
        self.target = torch.stack(self.target)
        
    def scale_fit_transform(self):
        new_features_dataset = pd.DataFrame(
            self.features_scaler.fit_transform(self.dataset.drop(columns=["Time", "Close"])),
            columns=self.dataset.drop(columns=["Time", "Close"]).columns
        )
        new_close_dataset = pd.DataFrame(
            self.scaler.fit_transform(self.dataset[["Close"]]),
            columns=["Close"]
        )

        self.dataset = pd.concat([new_features_dataset, new_close_dataset], axis=1)

        joblib.dump(self.features_scaler, os.path.join(self.scaler_path, "features_scaler.pkl"))
        joblib.dump(self.scaler, os.path.join(self.scaler_path, "scaler.pkl"))

    def scale_transform(self):
        new_features_dataset = pd.DataFrame(
            self.features_scaler.transform(self.dataset.drop(columns=["Time", "Close"])),
            columns=self.dataset.drop(columns=["Time", "Close"]).columns
        )
        new_close_dataset = pd.DataFrame(
            self.scaler.transform(self.dataset[["Close"]]),
            columns=["Close"]
        )

        self.dataset = pd.concat([new_features_dataset, new_close_dataset], axis=1)
