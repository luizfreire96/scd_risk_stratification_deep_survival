import os
import wfdb
import torch

import pandas as pd

from glob import glob

from torch.utils.data import DataLoader
from torchsurv.metrics.cindex import ConcordanceIndex

from utils.data import ECGDataset
from utils.models import CNN_LSTM_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

data_path = os.path.join("raw_data/Holter_ECG/")
results_path = "treated_data/ts_features/"

tab_data = pd.read_parquet("test_tab.parquet")

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
    if patient.split("/")[-1][:5] in list(tab_data.index.tolist())]

filtered_patients = []
for patient in patients:
    if wfdb.rdheader(patient).n_sig == 3:
        filtered_patients.append(patient)

patient_indexes = [i.split("/")[-1][:5] for i in filtered_patients]
labels = labels.loc[patient_indexes]

labels.loc[labels['Cause of death'] != 3, ['Cause of death']] = 0
labels.loc[labels['Cause of death'] == 3, ['Cause of death']] = 1

test_dataset = ECGDataset(ts_paths=filtered_patients,
                          tabular_data=tab_data, labels=labels, seq_len=max_len)
test_labels = [i.split("/")[-1][:5] for i in patient_indexes]
test_labels = labels.loc[test_labels]

dilation = [19, 1, 21]
kernel_size = 14
lstm_hidden_size = 128

model = CNN_LSTM_net(n_additional_features=0,
                     seq_len=max_len,
                     kernel_size=kernel_size,
                     dilation=dilation,
                     lstm_hidden_size=lstm_hidden_size,
                     stride=8)

state_dict = torch.load(
    "model_checkpoints/conv_lstm_index_survival_model/final_model_checkpoint_25.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.to(device)
model.eval()

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# auc com base teste

all_patients = []
all_preds = []
all_labels = []
all_times = []

with torch.no_grad():
    for batch in test_loader:
        (inputs, tab_inputs), labels, times, patient = batch

        inputs = inputs.to(device)
        tab_inputs = tab_inputs.to(device)
        labels = labels.to(device)
        times = times.to(device)

        outputs = model(inputs, tab_inputs).squeeze(-1)

        all_preds.extend(outputs)
        all_labels.extend(labels)
        all_times.extend(times)
        all_patients.extend(patient)

    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels).bool()
    all_times = torch.tensor(all_times)

cindex = ConcordanceIndex()
ci = cindex(all_preds, all_labels, all_times)

dl_hazard = pd.DataFrame({"Patient ID": all_patients,
                          "Cause of death": all_labels,
                          "days_4years": all_times,
                          "dl_hazard": all_preds})
dl_hazard.to_csv("z_dl_hazard.csv")
