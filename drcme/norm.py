import numpy as np
import pandas as pd
from sklearn import preprocessing

def normalize_ds(array1, norm_type, norm_mean=None):
    if norm_type == 1:
        #Scale to mean waveform
        scaler = preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(array1)
        #array1 = preprocessing.scale(array1, axis=1)
    elif norm_type == 2:
        array1 = preprocessing.scale(array1, axis=1)
        scaler = preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(array1)
    elif norm_type == 3:
        #manually Scale to mean waveform
        normalize = preprocessing.Normalizer(copy=False)
        scaler = preprocessing.StandardScaler(copy=False)
        np.nan_to_num(array1, copy=False)
        normalize.fit_transform(array1)
        scaler.fit_transform(array1)
    elif norm_type == 4:
        #Scale by min max within sample
        normalize = preprocessing.Normalizer(norm = 'l1', copy=False)
        array1 = normalize.fit_transform(array1)
        scaler = preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(array1)
    elif norm_type == 5:
        #Scale by min max within sample

        baseline = np.mean(array1[:,:20], axis=1).reshape(-1,1)
        array1 = array1 - baseline
        array1 = preprocessing.maxabs_scale(array1, axis=1, copy=False)
        scaler = preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(array1)
    elif norm_type == 6:
        baseline = np.mean(array1[:,:20], axis=1).reshape(-1,1)
        array1 = array1 - baseline
        array1 = preprocessing.maxabs_scale(array1, axis=1, copy=False)
        scaler = preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(array1)

    return array1
    

