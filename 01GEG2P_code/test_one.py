import os
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import islice
from concurrent.futures import ThreadPoolExecutor

from utils.DLGWAS_model import Net as DLGWAS
from utils.DNNGP_model import Net as DNNGP
from utils.DeepGS_model import Net as DeepGS
from utils.gMLP_Prox_tc import model as gmlp
from utils.LCNN_model import Net as LCNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class SNP_Dataset(Dataset):
    def __init__(self, phe, gene_id, snp):
        self.phe = phe
        self.gene_id = gene_id
        self.snp = snp

    def __getitem__(self, index):
        return self.gene_id[index], self.phe[index], self.snp[index]

    def __len__(self):
        return len(self.phe)


def get_dataloader(snp_path, phe_path, cvf_path, area, k):
    list_id, list_phe = [], []
    list_id_test, list_phe_test = [], []
    snp_test = []
    dictsnp = {}

    with open(snp_path) as file:
        for line in islice(file, 1, None):
            id = line.split(",")[0]  # Gene ID
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
    test_cvf = cvf[cvf['cv_1'] == k]
    for _, row in test_cvf.iterrows():
        id = row.iloc[0]
        list_id_test.append(id)
        list_phe_test.append([dict_phe[id]])
        snp_test.append(dictsnp[id])
    snp_test = np.array(snp_test, dtype=np.float32)
    list_phe_test = np.array(list_phe_test, dtype=np.float32)

    return list_id_test, list_phe_test, snp_test


def predict(plant,traits,models,snp_path, phe_path, cvf_path,device,kmax,snp_num):
    for k in range(1, kmax+1):
        for trait in traits:
            output_df_train = pd.DataFrame()
            output_df_test = pd.DataFrame()

            list_id_test, list_phe_test, snp_test = get_dataloader(snp_path, phe_path, cvf_path, trait, k)
            output_df_train['ID'] = list_id_test

            for name, model_class in models.items():
                # Determine if it's a PyTorch model class
                if isinstance(model_class, type) and issubclass(model_class, torch.nn.Module):
                    # PyTorch model class needs to be instantiated and moved to device
                    model = model_class(snp_num).to(device)
                    model_file = f'model/{plant}/k{k}/{trait}/{name}.pth'
                    model.load_state_dict(torch.load(model_file, map_location=device))
                    model.eval()
                    
                    all_preds_train = []
                    train_loader = DataLoader(
                        SNP_Dataset(torch.from_numpy(list_phe_test), list_id_test,
                                    torch.tensor(snp_test, dtype=torch.float32)),
                        batch_size=8, shuffle=False)

                    for _, _, snp_batch in train_loader:
                        snp_batch = snp_batch.to(device)
                        with torch.no_grad():
                            y_pred = model(snp_batch).cpu().numpy()
                            all_preds_train.append(y_pred)

                    output_df_train[name] = np.concatenate(all_preds_train, axis=0)
                else:
                    # sklearn and other models
                    model_file = f'model/{plant}/k{k}/{trait}/{name}.pkl'
                    model = joblib.load(model_file)
                    
                    y_pred_train = model.predict(snp_test)
                    output_df_train[name] = y_pred_train

            output_file_train = f'results/{plant}/k{k}/{trait}.csv'
            os.makedirs(os.path.dirname(output_file_train), exist_ok=True)
            if os.path.exists(output_file_train):

                existing_df = pd.read_csv(output_file_train)

                output_df_train = pd.concat([existing_df, output_df_train.drop(columns='ID')], axis=1)
            else:

                pass
            output_df_train.to_csv(output_file_train, index=False)
            print(f'Training results saved to {output_file_train}')
