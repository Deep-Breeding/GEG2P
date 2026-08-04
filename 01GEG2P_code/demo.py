import argparse
import pandas as pd
import os
import joblib
import optuna
import torch
import numpy as np
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge

from train_DL import train_DL
from train_ML import train_ML
# Predict within ten folds
from test_one import predict
from GEG2P import GEG2P_v1, GEG2P_v2, GEG2P_v3

# Deep learning models
from utils.LCNN_model import Net as LCNN
from utils.DLGWAS_model import Net as DLGWAS
from utils.DNNGP_model import Net as DNNGP
from utils.DeepGS_model import Net as DeepGS
from utils.gMLP_Prox_tc import model as gmlp


def main():
    parser = argparse.ArgumentParser(description="GEG2P pipeline runner")

    # Basic parameters
    parser.add_argument("--plant", type=str, required=True, help="Plant name, e.g. _Maize")
    parser.add_argument("--snp_path", type=str, required=True, help="Path to SNP file")
    parser.add_argument("--phe_path", type=str, required=True, help="Path to phenotype file")
    parser.add_argument("--cvf_path", type=str, required=True, help="Path to CVF file")
    parser.add_argument("--traits", type=str, nargs="+", required=True, help="Traits to predict, e.g. DTA PH KWPE")
    parser.add_argument("--kmax", type=int, default=1, help="Number of folds (default=1)")
    parser.add_argument("--snp_num", type=int, default=42938, help="Number of SNPs")

    # Control training steps
    parser.add_argument("--run_ML", action="store_true", help="Run machine learning models")
    parser.add_argument("--run_DL", action="store_true", help="Run deep learning models")
    parser.add_argument("--run_predict_ML", action="store_true", help="Run prediction step")
    parser.add_argument("--run_predict_DL", action="store_true", help="Run prediction step")
    parser.add_argument("--run_GEG2P", action="store_true", help="Run GEG2P ensemble models")

    args = parser.parse_args()

    plant = args.plant
    snp_path = args.snp_path
    phe_path = args.phe_path
    cvf_path = args.cvf_path
    traits = args.traits
    kmax = args.kmax
    snp_num = args.snp_num

    # ================================
    # Machine Learning models
    # ================================
    ML_models = {
        'KNN': KNeighborsRegressor(),
        'XGBoost': xgb.XGBRegressor(),
        'MLP': MLPRegressor(),
        'Random Forest': RandomForestRegressor(),
        'SVR': SVR(),
    }

    # ================================
    # Deep Learning models
    # ================================
    DL_models = {
        'DeepGS': DeepGS,
        'LCNN': LCNN,
        'gmlp': gmlp,
        'DNNGP': DNNGP,
        'DLGWAS': DLGWAS,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================================
    # Step 1: Train ML
    # ================================
    if args.run_ML:
        train_ML(plant, traits, ML_models, snp_path, phe_path, cvf_path, kmax=kmax)

    # ================================
    # Step 2: Train DL
    # ================================
    if args.run_DL:
        train_DL(plant, traits, DL_models, snp_path, phe_path, cvf_path, device, num_workers=4, kmax=kmax, snp_num=snp_num)

    # ================================
    # Step 3: Predict_ML
    # ================================
    if args.run_predict_ML:
        models_for_pred = {
            'KNN': KNeighborsRegressor(),
            'Random Forest': RandomForestRegressor(),
            'XGBoost': xgb.XGBRegressor(),
            'MLP': MLPRegressor(),
            'SVR': SVR(),
        }
        predict(plant, traits, models_for_pred, snp_path, phe_path, cvf_path, device, kmax=kmax, snp_num=snp_num)

    # ================================
    # Step 4: Predict_DL
    # ================================
    if args.run_predict_DL:
        models_for_pred = {
            'DeepGS': DeepGS,
            'LCNN': LCNN,
            'gmlp': gmlp,
            'DNNGP': DNNGP,
            'DLGWAS': DLGWAS,
        }
        predict(plant, traits, models_for_pred, snp_path, phe_path, cvf_path, device, kmax=kmax, snp_num=snp_num)

    # ================================
    # Step 5: GEG2P ensemble
    # ================================
    if args.run_GEG2P:
        model_ML = ['KNN', 'Random Forest', 'XGBoost', 'MLP', 'SVR']
        model_DL = ['DNNGP','DeepGS','DLGWAS','LCNN','gmlp']
        model_SS = ['BayesA','BayesB','BayesC','BL','BRR','RRBLUP','LASSO','SPLS','RR','BRNN']
        model_GEG2P = ["GEG2P(ML)","GEG2P(DL)","GEG2P(SS)","GEG2P(v1)"]

        model_v1 = model_DL + model_ML + model_SS
        model_v2 = model_v1 + model_GEG2P

        # v1
        GEG2P_v1([model_ML, model_DL, model_SS, model_v1, model_v2], model_GEG2P, plant, traits, phe_path, cvf_path, kmax=kmax)
        # v2
        GEG2P_v2(model_v2, model_GEG2P, plant, traits, phe_path, cvf_path, kmax=kmax)
        # v3
        GEG2P_v3(model_v2, "GEG2P(v3)", plant, traits, phe_path, cvf_path, kmax=kmax)


if __name__ == "__main__":
    main()
