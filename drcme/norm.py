import numpy as np
import pandas as pd
from sklearn import preprocessing

def normalize_ds(array1, norm_type):
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
        #scaler.fit_transform(array1)
    elif norm_type == 4:
        #Scale by min max within sample
        array1 = preprocessing.minmax_scale(array1, (-1,1), axis=1, copy=False)
    elif norm_type == 5:
        #Scale by min max within sample
        normalize = preprocessing.Normalizer(copy=False)

        baseline = np.mean(array1[:,:30], axis=1).reshape(-1,1)
        array1 = array1 - baseline
        array1 = preprocessing.minmax_scale(array1, (-1,1), axis=1, copy=False)
    return array1
