import pandas as pd
from itertools import islice


import numpy as np



def get_dataloader(snp_path, phe_path, cvf_path,  area):
    # total
    list_id = []
    list_phe = []
    # id
    list_id_test = []
    list_id_val = []
    list_id_train = []
    # phe
    list_phe_test = []
    list_phe_val = []
    list_phe_train = []
    # snp
    snp_test = []
    snp_val = []
    snp_train = []
    dictsnp = {}

    with open(snp_path) as file:
        id = ''
        for line in islice(file, 1, None):  # Iterate through each line starting from the second line (index 1)
            id = line.split(",")[0]  # gene id
            list_id.append(id)
            list_str = line.split(",")[1:32337]
            list_int = [float(x) for x in list_str]
            dictsnp[id] = list_int
    phe = open(phe_path, 'r')
    phe_data = pd.read_csv(phe)
    list_phe = phe_data[area]
    list_phe = list_phe.to_numpy()  # Convert DataFrame to NumPy array.
    list_phe = list_phe.reshape(-1, 1)  # Convert to a 2D array with 1 column
    cvf_file = open(cvf_path, 'r')
    cvf = pd.read_csv(cvf_file)
    # test_loader
    test_data = cvf[cvf['cv_1'] == 1].index

    for ht in range(int(len(test_data))):
        list_id_test.append(list_id[test_data[ht]])
        list_phe_test.append(list_phe[test_data[ht]])
        snp_test.append(dictsnp[list_id_test[ht]])
    list_phe_test = np.array(list_phe_test)
    snp_test = np.array(snp_test)
    mod_test = [snp_test, list_phe_test]

    train_data = cvf[(cvf['cv_1'] != 1)].index
    for ht in range(int(len(train_data))):
        list_id_train.append(list_id[train_data[ht]])
        list_phe_train.append(list_phe[train_data[ht]])
        snp_train.append(dictsnp[list_id_train[ht]])
    list_phe_train = np.array(list_phe_train)
    snp_train = np.array(snp_train)
    mod_train = [snp_train, list_phe_train]
    # return val_loader,test_loader,train_loader
    return mod_test, mod_train
