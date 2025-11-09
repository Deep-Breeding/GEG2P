# GEG2P Tutorial
This tutorial will guide you through running the GEG2P algorithm smoothly. With this tutorial, you only need three steps to complete the genomic prediction task.
## Step 1: Install Anaconda
1. It is recommended to use [Anaconda](https://www.anaconda.com/download) package management tool to build the model training environment. Please ensure your computer has an **NVIDIA graphics card** and the graphics driver is properly installed.  
2. Switch to domestic mirrors to prevent slow installation or errors when installing packages. Refer to [How to switch Anaconda to domestic sources](https://liguang.wang/index.php/archives/37/). You can choose Shanghai Jiao Tong University source or Tsinghua University source.
## Step 2: Create Environment
1. Enter the GEG2P directory  
```bash
cd GEG2P
```
2. Execute the following command to create the GEG2P environment
```bash
# Run in the GEG2P project path
chmod +x restore_env.sh
./restore_env.sh
```
This command will automatically call `GEG2P_env.yml` to create a conda environment and install Python packages and R language environment.  
If you need to modify the environment name, please change it synchronously in `GEG2P_env.yml` and `restore_env.sh`.  
## Step 3: Execute GEG2P Training and Prediction
1. Activate the GEG2P environment. Execute the following command:
```
conda activate GEG2P
```
2. Train deep learning base learners. Execute the following command:
```
bash run_all.sh
```

  Please modify the following parameters as needed:
1. `CONDA_ENV`: "Environment name"
2. `run_name`: "Folder name for saving models and prediction results"
3. `SNP_PATH`: "Genotype file path"
4. `PHE_PATH`: "Phenotype file path"
5. `CVF_PATH`: "Cross-validation partition file path"
6. `TRAITS`: "Traits, such as TRAITS=("DTA" "PH" "EH" "EL" "KWPE")"
7. `KMAX`: "Number of folds, get results of (1-KMAX) folds, default is 10"
9. `run_DL`: "Train deep learning models"
10. `run_predict_DL`: "Output prediction values to `results/${run_name}/k{1-10}/{trait}.csv`"  
11. `SNP_NUM`: "Number of SNPs, default is 42938"

Note that `run_name` only needs to provide the folder name for saving, no absolute path is needed, the code will automatically save it under the GEG2P folder.  
After completing GEG2P training, the following files can be obtained in the folder set by `run_name`:
```bash
- _Mazie_GA_process/
- GEG2P(DL)/
- GEG2P(ML)/
- GEG2P(SS)/
- GEG2P(v1)/
- GEG2P(v2)/
- GEG2P(v3)/
- k1/
- k2/
- k3/
- k4/
- k5/
- k6/
- k7/
- k8/
- k9/
- k10/
  ```


1. `_Mazie_GA_process`: "Optimal solutions (weight vectors) of each generation of the genetic algorithm"
2. `k1-k10`: "Prediction values from fold 1 to fold 10"
3. `GEG2P folders`: "Results of different ensemble strategies": Each folder contains the following files
```bash
    ├── 10_MSE.csv        # MSE of each trait in k-fold
    ├── 10_pcc.csv        # PCC of each trait in k-fold
    ├── 10_weights.csv    # Model weights of each trait in k-fold
    ├── avg_MSE.csv       # Average MSE of each trait
    ├── avg_pcc.csv       # Average PCC of each trait
    ├── avg_weights.csv   # Average weights of each trait
    ├── se_MSE.csv        # Standard error of MSE of each trait
    ├── se_pcc.csv        # Standard error of PCC of each trait
    └── se_weights.csv    # Standard error of weights of each trait
