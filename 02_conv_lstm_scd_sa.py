import os
import wfdb
import optuna
import json
import torch

import pandas as pd
import numpy as np
import torch.nn as nn

from datetime import datetime
from glob import glob
from optuna.storages import JournalStorage, JournalFileStorage

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import Subset
from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.cindex import ConcordanceIndex
from sklearn.model_selection import train_test_split, StratifiedKFold

from utils.training_models import EarlyStoppingMax
from utils.data import ECGDataset
from utils.models import CNN_LSTM_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

data_path = os.path.join("raw_data/Holter_ECG/")
results_path = "treated_data/ts_features/"

tab_data = pd.read_parquet(
    "treated_data/tabular_data/tabular_data_treated.parquet")

labels = tab_data[["Patient ID", "Cause of death", "days_4years"]].set_index("Patient ID")

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
    if patient.split("/")[-1][:5] in list(tab_data["Patient ID"])]

filtered_patients = []
for patient in patients:
    if wfdb.rdheader(patient).n_sig == 3:
        filtered_patients.append(patient)

patient_indexes = [i.split("/")[-1][:5] for i in filtered_patients]
labels = labels.loc[patient_indexes]

labels.loc[labels['Cause of death'] != 3, ['Cause of death']] = 0
labels.loc[labels['Cause of death'] == 3, ['Cause of death']] = 1

X_train, X_test, y_train, y_test = train_test_split(
    filtered_patients, labels, test_size=0.3, random_state=42, stratify=labels["Cause of death"])

# save patients to compare with tabular data
train_indexes = [i.split("/")[-1][:5] for i in X_train]
test_indexes = [i.split("/")[-1][:5] for i in X_test]

tab_model_df = tab_data.set_index("Patient ID")
tab_model_df.loc[train_indexes].to_parquet("train_tab.parquet")
tab_model_df.loc[test_indexes].to_parquet("test_tab.parquet")

tab_model_df = tab_model_df.drop(columns="Cause of death")

train_dataset = ECGDataset(ts_paths=X_train, tabular_data=tab_model_df, labels=labels, seq_len=max_len)
train_labels = [i.split("/")[-1][:5] for i in X_train]
train_labels = labels.loc[train_labels]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial: optuna.Trial):
    # Hiperparâmetros a serem testados
    global current_trial
    current_trial = trial.number
    step = 0
    writer = SummaryWriter(log_dir=f'runs/conv_lstm_index_model_survival_{current_trial}_{str(datetime.now())}')
    
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    dilation = [trial.suggest_int(f"dilation_{i}", 1, 25) for i in range(3)]
    kernel_size = 14
    lstm_hidden_size = 128
    
    num_epochs = 100

    trial_params = {"num_additional_features": 0,
                    "input_channels": 3,
                    "stride": 8,
                    "dilation": dilation,
                    "kernel_size": kernel_size,
                    "lstm_hidden_size": lstm_hidden_size}
    with open(f"model_checkpoints/conv_lstm_index_survival_model/{current_trial}_trial_params.json", "w") as f:
        json.dump(trial_params, f, indent=4)

    global_epoch = 0

    mean_ci = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(train_labels.shape[0]), train_labels["Cause of death"])):
        model = CNN_LSTM_net(n_additional_features=0,
                             seq_len=max_len,
                             kernel_size=kernel_size,
                             dilation=dilation,
                             lstm_hidden_size=lstm_hidden_size,
                             stride=8).to(device)
        model = nn.DataParallel(model)
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-5)
    
        train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=64, shuffle=True)
        val_loader   = DataLoader(Subset(train_dataset, val_idx), batch_size=64, shuffle=True)
        early_stopping = EarlyStoppingMax(patience=5)
    
        for epoch in range(num_epochs):
            model.train()
            running_loss = torch.tensor(0.0)
            running_loss = running_loss.to(device)
            progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/{num_epochs}")
        
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
                loss = neg_partial_log_likelihood(outputs, labels, times, reduction="mean")
    
                loss.backward()
                optimizer.step()
    
                running_loss += loss.detach()
    
                progress_bar.set_postfix(loss=[running_loss/(batch_idx+1), loss])
    
                step = step + 1
                writer.add_scalar("Current Loss/train", loss, step)
    
            # auc com base teste
            model.eval()
            all_preds = []
            all_labels = []
            all_times = []
    
            with torch.no_grad():
                for batch in val_loader:
                    (inputs, tab_inputs), labels, times, patient = batch
    
                    inputs = inputs.to(device)
                    tab_inputs = tab_inputs.to(device)
                    labels = labels.to(device)
                    times = times.to(device)
    
                    outputs = model(inputs, tab_inputs).squeeze(-1)
    
                    all_preds.extend(outputs)
                    all_labels.extend(labels)
                    all_times.extend(times)
                    
                all_preds = torch.tensor(all_preds)
                all_labels = torch.tensor(all_labels).bool()
                all_times = torch.tensor(all_times)
    
            cindex = ConcordanceIndex()
            ci = cindex(all_preds, all_labels, all_times)
            
            early_stopping(ci.item(), model, path=f'model_checkpoints/conv_lstm_index_survival_model/{fold}_{current_trial}_best_model_checkpoint.pth')
            if early_stopping.early_stop:
                print(f"Early Stopping na época {epoch}!")
                break
            global_epoch = global_epoch + 1
            torch.save(model,
                       f'model_checkpoints/conv_lstm_index_survival_model/model_checkpoint_{fold}_{current_trial}_{global_epoch}.pth')
    
            # log
            writer.add_scalar("Epoch Loss/train", running_loss, global_epoch)
            writer.add_scalar("ci/val", ci, global_epoch)
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("Learning Rate", lr, global_epoch)
        
        if early_stopping.best_score < 0.5:
            ci = 0
        else:
            ci = early_stopping.best_score
        mean_ci.append(ci)

    print("Treinamento concluído!")
    writer.add_scalar("mean_best_ci/val", np.mean(mean_ci), current_trial)
    return np.mean(mean_ci)


storage = JournalStorage(JournalFileStorage("optuna_study_conv_lstm_survival.log"))

study = optuna.create_study(
    study_name="optuna_study_conv_lstm_survival",
    storage=storage,
    load_if_exists=True,
    direction="maximize")
study.optimize(objective, n_trials=100, gc_after_trial=True)


# Acessar todos os trials
trials_df = study.trials_dataframe()
print(trials_df)

# Acessar um trial específico (ex: o melhor)
best_trial = study.best_trial
print(f"Melhor trial: {best_trial.number}, Parâmetros: {best_trial.params}, Valor: {best_trial.value}")

best_params = study.best_params
with open(f"model_checkpoints/conv_lstm_index_survival_model/{best_trial.number}_best_trial_params.json", "w") as f:
    json.dump(best_params, f, indent=4)
