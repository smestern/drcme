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
        array1 = preprocessing.maxabs_scale(array1, axis=1, copy=False)
    elif norm_type == 5:
        #Scale by min max within sample

        baseline = np.mean(array1[:,:20], axis=1).reshape(-1,1)
        array1 = array1 - baseline
        array1 = preprocessing.maxabs_scale(array1, axis=1, copy=False)
        #scaler = preprocessing.StandardScaler(copy=False)
        #scaler.fit_transform(array1)
    elif norm_type == 6:
        #baseline = np.mean(array1[:,:20], axis=1).reshape(-1,1)
        #array1 = array1 - baseline
        array1 = preprocessing.maxabs_scale(array1, axis=1, copy=False)
        scaler = preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(array1)

    return array1
    
def center_on_m(array1, array2, cntr='min'):
    if cntr == 'min':
       a1m = np.nanmean(array1, axis=0)
       a2m = np.nanmean(array2, axis=0)
       a1m = np.argmin(a1m)
       a2m = np.argmin(a2m)
       subm = a1m-a2m
       ### First set the values to equal
       if subm > 0:
           array1 = array1[:,int(subm):]
       elif subm < 0:
           array2 = array2[:,int(subm):]

    return array1, array2

def equal_ar_size(array1, array2):
    a1 = array1.shape[1]
    a2 = array2.shape[1]
    if a1 > a2:
           array1 = array1[:,:(-1*int(a1-a2))]
    elif a2 > a1:
          array2 = array2[:,:(-1*int(a2-a1))]
    return array1, array2

def shift_means(array1, array2, axis=0):
    a1m = np.mean(array1, axis=axis)
    a2m = np.mean(array2, axis=axis)
    subm = a1m - a2m
    array2_shift = array2 - subm
    return array1, array2_shift