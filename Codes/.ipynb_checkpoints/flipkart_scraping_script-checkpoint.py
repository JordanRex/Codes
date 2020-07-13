import sys

import pandas as pd, numpy as np
from collections import OrderedDict
import re
from pandas.io.json import json_normalize
import Levenshtein
from sklearn.feature_selection import VarianceThreshold

# removing near zero variance columns
def variance_threshold_selector(train, threshold):
    selector = VarianceThreshold(threshold)
    selector.fit(train)
    X = train[train.columns[selector.get_support(indices=True)]]
#     print(f'features kicked: {len(train.columns) - len(X.columns)}')
    return X

# correlation function
def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
    return dataset

## MAPE function
def mape(a, b): 
    mask = a != 0
    return (np.fabs(a - b)/a)[mask].mean() * 100

## json normalize product des
def jsonnorm_prod(df):
    df['x'] = df['x'].str.replace(r' "', '"').astype(str)
    df['x'] = df['x'].str.findall(r'("[a-zA-Z0-9 ]+": "[a-zA-Z0-9 ]+")').astype(str)
    df['x'] = df['x'].str.replace(r'[', '{').str.replace(r']', '}').str.replace("'", "").astype(str)
    df['x'] = df['x'].map(eval)
    df['x'] = df['x'].str.replace(r'[a-zA-Z0-9 ]+": "[a-zA-Z0-9 ]+', '').str.replace(r'"",,', '').astype(str)

    temp = json_normalize(df['x'])

    temp = temp.add_prefix('attr_')
    temp.rename(columns=lambda x: x.strip(), inplace=True)
    temp = temp.fillna('.').groupby(temp.columns, axis=1).max()
    
    df = pd.concat([df.reset_index(drop=True), temp], axis=1)
    df = df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('.'))
    df = df.replace('nan', '.')
    return df

# function to replace some string and get extract for comparison with base data
def match_product_title(obj_str):
    extracted = [i for i in brands if i in obj_str]
    if len(extracted)>0:
        extracted = max(extracted, key=len)
    else:
        extracted = ''
    return extracted

if __name__ == "__main__":
    arg1 = sys.argv[1]
    main(arg1)