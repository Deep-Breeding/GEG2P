import argparse
import pandas as pd
import os
import sys
import joblib
import torch
import numpy as np
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Add current directory to path for module import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Predict within k-fold cross-validation
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

    # Control prediction and ensemble steps
    parser.add_argument("--run_predict_ML", action="store_true", help="Run ML prediction step")
    parser.add_argument("--run_predict_DL", action="store_true", help="Run DL prediction step")
    parser.add_argument("--run_GEG2P", action="store_true", help="Run GEG2P ensemble models")

    args = parser.parse_args()

    plant = args.plant
    snp_path = args.snp_path
    phe_path = args.phe_path
    cvf_path = args.cvf_path
    traits = args.traits
    kmax = args.kmax
    snp_num = args.snp_num

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================================
    # Step 3: Predict_ML
    # ================================
    if args.run_predict_ML:
        models_for_pred = {
            'KNN': KNeighborsRegressor(),
            'RandomForest': RandomForestRegressor(),
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
            'LCNN': LCNN,
            'gmlp': gmlp,
            'DNNGP': DNNGP,
            'DLGWAS': DLGWAS,
            'DeepGS': DeepGS,
        }
        predict(plant, traits, models_for_pred, snp_path, phe_path, cvf_path, device, kmax=kmax, snp_num=snp_num)

    # ================================
    # Step 5: GEG2P ensemble
    # ================================
    if args.run_GEG2P:
        model_ML = ['KNN', 'RandomForest', 'XGBoost', 'MLP', 'SVR']
        model_DL = ['DNNGP','DeepGS','DLGWAS','LCNN','gmlp']
        model_SS = ['BayesA','BayesB','BayesC','BL','BRR','RRBLUP','LASSO','SPLS','RR','BRNN']
        # model_SS = ['BayesA','BayesB','BayesC','BL','BRR','RRBLUP','LASSO','SPLS','RR']
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
