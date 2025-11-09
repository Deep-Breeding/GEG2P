# -*- coding: utf-8 -*-
import os
from itertools import islice
import joblib
import optuna
import pandas as pd
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def get_dataloader(snp_path, phe_path, cvf_path, area, k):
    list_id, list_phe = [], []
    list_id_test, list_id_train = [], []
    list_phe_test, list_phe_train = [], []
    snp_test, snp_train = [], []
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
    dict_phe = dict(zip(phe_data.iloc[:, 0], phe_data[area]))

    # Read cross-validation grouping information
    cvf = pd.read_csv(cvf_path)

    # Split data according to information in cvf, only filter data existing in cvf, ignore data not existing in cvf
    # Test set
    test_cvf = cvf[cvf['cv_1'] == k]
    for _, row in test_cvf.iterrows():
        id = row.iloc[0]  # First column is ID
        list_id_test.append(id)
        list_phe_test.append([dict_phe[id]])  # Find phenotype by ID
        snp_test.append(dictsnp[id])  # Find genotype by ID
    mod_test = [np.array(snp_test), np.array(list_phe_test)]

    # Training set
    train_cvf = cvf[cvf['cv_1'] != k]
    for _, row in train_cvf.iterrows():
        id = row.iloc[0]  # First column is ID
        list_id_train.append(id)
        list_phe_train.append([dict_phe[id]])
        snp_train.append(dictsnp[id])
    mod_train = [np.array(snp_train), np.array(list_phe_train)]

    return mod_test, mod_train, list_id_test

def objective(trial, X_train, y_train, X_test, y_test, model_name):
    if model_name == 'KNN':
        n_neighbors = trial.suggest_int('n_neighbors', 3, 15)
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
    elif model_name == 'Random Forest':
        max_depth = trial.suggest_int('max_depth', 10, 50)
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    elif model_name == 'XGBoost':
        max_depth = trial.suggest_int('max_depth', 3, 10)
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        model = xgb.XGBRegressor(max_depth=max_depth, n_estimators=n_estimators)
    elif model_name == 'MLP':
        hidden_size1 = trial.suggest_int('hidden_size1', 10, 100)
        hidden_size2 = trial.suggest_int('hidden_size2', 10, 100)
        alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
        hidden_layer_sizes = (hidden_size1, hidden_size2)
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha)
    elif model_name == 'Bayesian Ridge':
        alpha_1 = trial.suggest_loguniform('alpha_1', 1e-7, 1e-1)
        alpha_2 = trial.suggest_loguniform('alpha_2', 1e-7, 1e-1)
        lambda_1 = trial.suggest_loguniform('lambda_1', 1e-7, 1e-1)
        lambda_2 = trial.suggest_loguniform('lambda_2', 1e-7, 1e-1)
        model = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2)
    elif model_name == 'SVR':
        C = trial.suggest_loguniform('C', 1e-3, 1e2)
        cache_size = 2000
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        model = SVR(C=C, kernel=kernel, cache_size=cache_size, verbose=True, shrinking=False)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pcc, _ = pearsonr(y_test, y_pred)
    return pcc


def train_model(plant,trait, model_name,snp_path, phe_path, cvf_path,kmax):
    print(f'Training {model_name} for trait {trait}')
    test_loader, train_loader, list_id_test = get_dataloader(snp_path, phe_path, cvf_path, trait, 1)
    X_train, y_train = train_loader[0], train_loader[1]
    X_test, y_test = test_loader[0], test_loader[1]
    y_train = y_train.ravel()
    y_test = y_test.ravel()


    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, model_name), n_trials=20, timeout=10800)
    best_params = study.best_params
    if model_name == 'KNN':
        model = KNeighborsRegressor(**best_params)
    elif model_name == 'Random Forest':
        model = RandomForestRegressor(**best_params)
    elif model_name == 'XGBoost':
        model = xgb.XGBRegressor(**best_params)
    elif model_name == 'MLP':
        hidden_layer_sizes = (best_params['hidden_size1'], best_params['hidden_size2'])
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, alpha=best_params['alpha'])
    elif model_name == 'SVR':
        print("starting optimization of SVR...")
        model = SVR(**best_params)
    else:
        model = BayesianRidge(**best_params)
    for k in range(1, kmax+1):
        b_model=model
        test_loader, train_loader, list_id_test = get_dataloader(snp_path, phe_path, cvf_path, trait, k)
        X_train, y_train = train_loader[0], train_loader[1]
        X_test, y_test = test_loader[0], test_loader[1]
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        print(f'Training {model_name} for trait {trait}\nk:{k}')

        b_model.fit(X_train, y_train)
        y_pred = b_model.predict(X_test)
        pcc, _ = pearsonr(y_test, y_pred)

        model_filename = f'model/{plant}/k{k}/{trait}/{model_name}.pkl'
        directory = os.path.dirname(model_filename)
        os.makedirs(directory, exist_ok=True)
        joblib.dump(b_model, model_filename)
        print(f'Model {model_name} for trait {trait}, fold {k} :pcc:{pcc}, best params: {best_params}.')

# Parallel version
def train_ML_bx(plant,traits,models,snp_path, phe_path, cvf_path,kmax):
    for trait in traits:
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(train_model,plant, trait,  name,snp_path, phe_path, cvf_path,kmax): name for name in models.keys()}
            for future in futures:
                future.result()

# Sequential training version
def train_ML(plant,traits,models,snp_path, phe_path, cvf_path,kmax):
    for trait in traits:
        for name in models.keys():
            print(f'Training {name} for trait {trait}')
            train_model(plant, trait, name, snp_path, phe_path, cvf_path, kmax)
