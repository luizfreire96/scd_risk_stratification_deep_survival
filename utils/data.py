import numpy as np
import pandas as pd
import torch
import wfdb
import math
import os

from torch.utils.data import Dataset
import torch.nn.init as init


def read_wfdb(record_path):
    record = wfdb.rdrecord(record_path)
    signals = record.p_signal
    channels = record.sig_name
    fs = record.fs
    
    time = np.arange(signals.shape[0]) / fs
    df = pd.DataFrame(signals, columns=channels)
    df.insert(0, 'timestamp', time)
    return df


class ECGDataset(Dataset):
    def __init__(self, ts_paths, tabular_data, labels, seq_len):
        
        self.tabular_data = tabular_data
        self.ts_paths = ts_paths
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ts_paths)

    def __getitem__(self, idx):
                    
        # Carrega a serie
        ts_path = self.ts_paths[idx]
        ts_df = read_wfdb(ts_path).drop(columns="timestamp").to_numpy()
        ts = torch.FloatTensor(ts_df)
        ts = ts.permute(1, 0)
        
        repeats = math.ceil(self.seq_len / ts.size()[1])
        ts = ts.repeat(1, repeats)
        ts = ts[:, -self.seq_len:]

        patient = ts_path.split("/")[-1]
        
        # Pega dados tabulares
        tab_row = self.tabular_data.loc[patient[:5]].to_numpy()
        tab_row = torch.tensor(tab_row).float()
        
        # Pega o rótulo
        label = self.labels.loc[patient[:5], "Cause of death"]
        label = torch.tensor(label)
        label = label.squeeze()
        
        time = self.labels.loc[patient[:5]]["days_4years"]
        time = torch.tensor(time).float()
        time = time.squeeze()

        return (ts, tab_row), label, time, patient
