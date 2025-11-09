import os
import re
from itertools import islice

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader

from utils.LCNN_model import Net as LCNN
from utils.DLGWAS_model import Net as DLGWAS
from utils.DNNGP_model import Net as DNNGP
from utils.DeepGS_model import Net as DeepGS
from utils.gMLP_Prox_tc import model as gmlp


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class FedDataset(Dataset):
    def __init__(self, phe, gene_id, snp):
        self.phe = phe
        self.gene_id = gene_id
        self.snp = snp

    def __getitem__(self, index):
        return self.gene_id[index], self.phe[index], self.snp[index]

    def __len__(self):
        return len(self.phe)


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, patience, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, delta=0.01)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for gene_ids, phenotypes, snps in train_loader:
            phenotypes = phenotypes.float().to(device)
            snps = snps.float().to(device)
            optimizer.zero_grad()

            outputs = model(snps)
            loss = criterion(outputs, phenotypes)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validate the model
        val_loss, pcc = validate_model(model, val_loader, criterion, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, PCC: {pcc:.4f}")

        # Check for early stopping
        if early_stopping(val_loss):
            print("Early stopping triggered.")
            break

    return val_loss, pcc


def validate_model(model, val_loader, criterion, device='cpu'):
    model.eval()
    val_loss = 0.0
    predictions = []
    true_values = []

    with torch.no_grad():
        for gene_ids, phenotypes, snps in val_loader:
            phenotypes = phenotypes.float().to(device)
            snps = snps.float().to(device)
            outputs = model(snps)

            loss = criterion(outputs, phenotypes)
            val_loss += loss.item()

            predictions.extend(outputs.cpu().numpy())
            true_values.extend(phenotypes.cpu().numpy())

    predictions = np.array(predictions).ravel()
    true_values = np.array(true_values).ravel()
    pcc, _ = pearsonr(true_values, predictions)

    return val_loss / len(val_loader), pcc


def predict(model, test_loader, device):
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for gene_ids, phenotypes, snps in test_loader:
            snps = snps.float().to(device)
            outputs = model(snps)
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(phenotypes.cpu().numpy())

    predictions = np.array(predictions).ravel()
    true_values = np.array(true_values).ravel()

    mse = mean_squared_error(true_values, predictions)

    pcc, _ = pearsonr(true_values, predictions)

    return predictions, mse, pcc


def create_objective(model_class, trait,snp_path, phe_path, cvf_path,device,num_workers,snp_num):
    def objective(trial):
        model = model_class(snp_num)

        batch_size = trial.suggest_int("batch_size", 8, 64, step=8)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        num_epochs = trial.suggest_int('num_epochs', 20, 50)
        patience = trial.suggest_int('patience', 2, 10, step=2)

        train_loader, val_loader, _ = DL_dataloader(snp_path, phe_path, cvf_path, batch_size, trait,1,num_workers)
        val_loss, pcc = train_model(model, train_loader, val_loader, num_epochs, learning_rate, patience, device)

        print(f"Trial completed with PCC: {pcc:.4f}")

        return val_loss

    return objective


def DL_dataloader(snp_path, phe_path, cvf_path, batch_size, area, k, num_workers):
    list_id = []
    dictsnp = {}

    # Read SNP data
    with open(snp_path) as file:
        for line in islice(file, 1, None):
            id = line.split(",")[0]
            list_id.append(id)
            list_str = line.split(",")[1:]
            list_int = [float(x) for x in list_str]
            dictsnp[id] = list_int

    # Read phenotype data
    phe_data = pd.read_csv(phe_path)
    # Create ID to phenotype mapping
    dict_phe = dict(zip(phe_data.iloc[:, 0], phe_data[area].astype(np.float32)))

    # Read cross-validation grouping information
    cvf = pd.read_csv(cvf_path)
    k_val = k + 1
    if k == 10:
        k_val = 1

    # Split data according to information in cvf
    # Training set
    train_cvf = cvf[(cvf['cv_1'] != k) & (cvf['cv_1'] != k_val)]
    list_id_train, list_phe_train, snp_train = [], [], []
    for _, row in train_cvf.iterrows():
        id = row.iloc[0]  # First column is ID
        list_id_train.append(id)
        list_phe_train.append([dict_phe[id]])  # Filter phenotype by ID
        snp_train.append(dictsnp[id])  # Filter genotype by ID

    # Validation set
    val_cvf = cvf[cvf['cv_1'] == k_val]
    list_id_val, list_phe_val, snp_val = [], [], []
    for _, row in val_cvf.iterrows():
        id = row.iloc[0]  # First column is ID
        list_id_val.append(id)
        list_phe_val.append([dict_phe[id]])
        snp_val.append(dictsnp[id])

    # Test set
    test_cvf = cvf[cvf['cv_1'] == k]
    list_id_test, list_phe_test, snp_test = [], [], []
    for _, row in test_cvf.iterrows():
        id = row.iloc[0]  # First column is ID
        list_id_test.append(id)
        list_phe_test.append([dict_phe[id]])
        snp_test.append(dictsnp[id])

    # Build training data loader
    list_phe_train = torch.from_numpy(np.array(list_phe_train, dtype=np.float32))
    snp_train = torch.from_numpy(np.array(snp_train, dtype=np.float32))
    mdata_train = FedDataset(list_phe_train, list_id_train, snp_train)
    train_loader = DataLoader(dataset=mdata_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    # Build validation data loader
    list_phe_val = torch.from_numpy(np.array(list_phe_val, dtype=np.float32))
    snp_val = torch.from_numpy(np.array(snp_val, dtype=np.float32))
    mdata_val = FedDataset(list_phe_val, list_id_val, snp_val)
    val_loader = DataLoader(dataset=mdata_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Build test data loader
    list_phe_test = torch.from_numpy(np.array(list_phe_test, dtype=np.float32))
    snp_test = torch.from_numpy(np.array(snp_test, dtype=np.float32))
    mdata_test = FedDataset(list_phe_test, list_id_test, snp_test)
    test_loader = DataLoader(dataset=mdata_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader


def train_DL(plant,traits,model_classes,snp_path, phe_path, cvf_path,device,num_workers,kmax, snp_num):
    for trait in traits:
        print("trait:", trait)
        best_trials = {}

        for model_name, model_class in model_classes.items():
            print("train model:", model_name)
            study = optuna.create_study(direction='minimize')
            study.optimize(create_objective(model_class, trait,snp_path, phe_path, cvf_path,device,num_workers,snp_num), n_trials=50)

            best_trials[model_name] = study.best_trial

            best_params = best_trials[model_name].params

            batch_size, num_epochs, learning_rate, patience = best_params['batch_size'], best_params['num_epochs'], \
                best_params['learning_rate'], best_params['patience']
            # batch_size, num_epochs, learning_rate, patience = 8, 50, 1e-5, 7


            for k in range(1, kmax+1):
                best_model = model_class(snp_num)
                train_loader, val_loader, test_loader = DL_dataloader(snp_path, phe_path, cvf_path,
                                                                      batch_size, trait,k,num_workers)

                train_model(best_model, train_loader, val_loader, num_epochs, learning_rate,
                            patience, device)
                predictions, mse, pcc = predict(best_model, test_loader, device)


                model_file = f'model/{plant}/k{k}/{trait}/{model_name}.pth'

                directory = os.path.dirname(model_file)

                os.makedirs(directory, exist_ok=True)

                torch.save(best_model.state_dict(), model_file)
                '''
                output_file = f'model{plant}/k{k}/{trait}.txt'

                directory = os.path.dirname(output_file)

                os.makedirs(directory, exist_ok=True)

                with open(output_file, 'a') as file:
                    file.write(f'Model Name: {model_name}\n')

                    num_epochs = best_trials[model_name].params['num_epochs']
                    bs = best_trials[model_name].params['batch_size']
                    learning_rate = best_trials[model_name].params['learning_rate']
                    patience = best_trials[model_name].params['patience']
                    file.write(f'Best Parameters:\n')
                    file.write(f'Best batch-size:{bs}\n')
                    file.write(f'Num Epochs: {num_epochs}\n')
                    file.write(f'Learning Rate: {learning_rate}\n')
                    file.write(f'Patience: {patience}\n')
                    file.write(f'pcc: {pcc}\n')
                    file.write(f'MSE: {mse}\n')
'''