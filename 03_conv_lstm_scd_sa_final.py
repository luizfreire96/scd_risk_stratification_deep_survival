import os
import wfdb
import torch

import pandas as pd
import torch.nn as nn

from datetime import datetime
from glob import glob

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchsurv.loss.cox import neg_partial_log_likelihood

from utils.training_models import EarlyStoppingMax
from utils.data import ECGDataset
from utils.models import CNN_LSTM_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

data_path = os.path.join("raw_data/Holter_ECG/")
results_path = "treated_data/ts_features/"

tab_data = pd.read_parquet("train_tab.parquet")

labels = tab_data[["Cause of death", "days_4years"]]

patients = glob(data_path + "*.dat")
patients = [i.split(".")[0] for i in patients]

max_len = 0
for i in patients:
    header = wfdb.rdheader(i)
    if header.sig_len > max_len:
        max_len = header.sig_len

patients = [
    patient
    for patient in patients
    if patient.split("/")[-1][:5] in list(tab_data.index.tolist())
]

filtered_patients = []
for patient in patients:
    if wfdb.rdheader(patient).n_sig == 3:
        filtered_patients.append(patient)

patient_indexes = [i.split("/")[-1][:5] for i in filtered_patients]
labels = labels.loc[patient_indexes]

labels.loc[labels["Cause of death"] != 3, ["Cause of death"]] = 0
labels.loc[labels["Cause of death"] == 3, ["Cause of death"]] = 1

train_dataset = ECGDataset(
    ts_paths=filtered_patients, tabular_data=tab_data, labels=labels, seq_len=max_len
)
train_labels = [i.split("/")[-1][:5] for i in patient_indexes]
train_labels = labels.loc[train_labels]

step = 0
writer = SummaryWriter(
    log_dir=f"runs/final_conv_lstm_index_model_survival_{str(datetime.now())}"
)

lr = 7.247e-4

dilation = [3, 1, 23]
kernel_size = 14
lstm_hidden_size = 128

num_epochs = 100

model = CNN_LSTM_net(
    n_additional_features=0,
    seq_len=max_len,
    kernel_size=kernel_size,
    dilation=dilation,
    lstm_hidden_size=lstm_hidden_size,
    stride=8,
).to(device)

model = nn.DataParallel(model)

optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-4)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
early_stopping = EarlyStoppingMax(patience=10)

for epoch in range(num_epochs):
    running_loss = torch.tensor(0.0)
    running_loss = running_loss.to(device)
    progress_bar = tqdm(train_loader, desc=f"Época {epoch + 1}/{num_epochs}")

    for batch_idx, batch in enumerate(progress_bar):
        (inputs, tab_inputs), labels, times, patient = batch

        inputs = inputs.to(device)
        tab_inputs = tab_inputs.to(device)
        labels = labels.to(device)
        times = times.to(device)

        if labels.sum() == 0:
            continue

        optimizer.zero_grad()

        outputs = model(inputs, tab_inputs).squeeze(-1)
        loss = neg_partial_log_likelihood(
            outputs, labels, times, reduction="mean")

        loss.backward()
        optimizer.step()

        running_loss += loss.detach()

        progress_bar.set_postfix(loss=[running_loss / (batch_idx + 1), loss])

        step = step + 1
        writer.add_scalar("Current Loss/train", loss, step)

    early_stopping(
        -running_loss.item(),
        model,
        path="model_checkpoints/conv_lstm_index_survival_model/final_best_model_checkpoint.pth",
    )
    if early_stopping.early_stop:
        print(f"Early Stopping na época {epoch}!")
        break
    torch.save(
        model.module.state_dict(),
        f"model_checkpoints/conv_lstm_index_survival_model/final_model_checkpoint_{epoch}.pth",
    )

    # log
    writer.add_scalar("Epoch Loss/train", running_loss, epoch)
