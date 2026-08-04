#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for training a single model, used for parallel calling by shell scripts
"""

import argparse
import os
import sys
import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_ML import train_model as train_ml_model
from train_DL import train_DL

def setup_cpu_limits():
    """Set CPU core limits"""
    import os
    max_cores = os.environ.get('MAX_CPU_CORES', '0')
    try:
        max_cores = int(max_cores)
        if max_cores > 0:
            torch.set_num_threads(max_cores)
            # NumPy旧版本有set_num_threads，新版本使用环境变量OMP_NUM_THREADS
            try:
                if hasattr(np, 'set_num_threads'):
                    np.set_num_threads(max_cores)
            except AttributeError:
                pass
            os.environ['OMP_NUM_THREADS'] = str(max_cores)
            print(f'[PID {os.getpid()}] CPU core limit: {max_cores}')
        else:
            print(f'[PID {os.getpid()}] CPU core limit: No limit')
    except ValueError:
        print(f'[PID {os.getpid()}] 无效的MAX_CPU_CORES值，使用默认设置')

def train_single_ml(plant, trait, model_name, snp_path, phe_path, cvf_path, kmax):
    """Train a single ML model"""
    print(f'[PID {os.getpid()}] 开始训练 ML 模型: {model_name}, trait: {trait}')
    train_ml_model(plant, trait, model_name, snp_path, phe_path, cvf_path, kmax)
    print(f'[PID {os.getpid()}] 完成 ML 模型: {model_name}, trait: {trait}')

def train_single_dl(plant, trait, model_name, snp_path, phe_path, cvf_path, device, num_workers, kmax, snp_num):
    """Train a single DL model"""
    print(f'[PID {os.getpid()}] 开始训练 DL 模型: {model_name}, trait: {trait}')
    from utils.LCNN_model import Net as LCNN
    from utils.DLGWAS_model import Net as DLGWAS
    from utils.DNNGP_model import Net as DNNGP
    from utils.DeepGS_model import Net as DeepGS
    from utils.gMLP_Prox_tc import model as gmlp

    DL_models = {
        'LCNN': LCNN,
        'gmlp': gmlp,
        'DNNGP': DNNGP,
        'DLGWAS': DLGWAS,
        'DeepGS': DeepGS,
    }

    if model_name not in DL_models:
        raise ValueError(f"Unknown DL model: {model_name}")

    model_classes = {model_name: DL_models[model_name]}
    train_DL(plant, [trait], model_classes, snp_path, phe_path, cvf_path, device, num_workers, kmax, snp_num)
    print(f'[PID {os.getpid()}] 完成 DL 模型: {model_name}, trait: {trait}')

def main():
    parser = argparse.ArgumentParser(description="Train a single model")
    parser.add_argument("--plant", type=str, required=True, help="Plant name")
    parser.add_argument("--trait", type=str, required=True, help="Trait name")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--model_type", type=str, required=True, choices=['ML', 'DL'], help="Model type: ML or DL")
    parser.add_argument("--snp_path", type=str, required=True, help="Path to SNP file")
    parser.add_argument("--phe_path", type=str, required=True, help="Path to phenotype file")
    parser.add_argument("--cvf_path", type=str, required=True, help="Path to CVF file")
    parser.add_argument("--kmax", type=int, default=1, help="Number of folds")
    parser.add_argument("--device", type=str, default="cuda", help="Device for DL models")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DL dataloader")
    parser.add_argument("--snp_num", type=int, default=42938, help="Number of SNPs")

    args = parser.parse_args()
    
    # Set CPU core limits
    setup_cpu_limits()
    
    if args.model_type == 'ML':
        train_single_ml(args.plant, args.trait, args.model_name,
                       args.snp_path, args.phe_path, args.cvf_path, args.kmax)
    else:
        train_single_dl(args.plant, args.trait, args.model_name,
                       args.snp_path, args.phe_path, args.cvf_path,
                       args.device, args.num_workers, args.kmax, args.snp_num)

if __name__ == "__main__":
    main()
