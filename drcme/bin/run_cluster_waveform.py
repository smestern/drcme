import numpy as np
import pandas as pd
import drcme.spca_fit as sf
import drcme.load_data as ld
import argschema as ags
import joblib
import logging
import os
import json
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from scipy import signal
from sklearn.ensemble import IsolationForest
import math
output_fld = "output\\debug\\"
class DatasetParameters(ags.schemas.DefaultSchema):
    fv_h5_file = ags.fields.InputFile(description="HDF5 file with feature vectors")
    metadata_file = ags.fields.InputFile(description="Metadata file in CSV format", allow_none=True, default=None)
    
    dendrite_type = ags.fields.String(default="all", validate=lambda x: x in ["all", "spiny", "aspiny"])
    allow_missing_structure = ags.fields.Boolean(required=False, default=False)
    allow_missing_dendrite = ags.fields.Boolean(required=False, default=False)
    limit_to_cortical_layers = ags.fields.List(ags.fields.String, default=[], cli_as_single_argument=True)
    id_file = ags.fields.InputFile(description="Text file with IDs to use",
        required=False, allow_none=True, default=None)


class AnalysisParameters(ags.ArgSchema):
    params_file = ags.fields.InputFile(default="C://Users//SMest//source//repos//drcme//drcme//bin//default_spca_params.json")
    output_dir = ags.fields.OutputDir(description="directory for output files")
    output_code = ags.fields.String(description="code for output files")
    norm_type = ags.fields.Integer(default=0)
    labels_file = ags.fields.InputFile(description="label files", allow_none=True, default=None)
    datasets = ags.fields.Nested(DatasetParameters,
                                 required=True,
                                 many=True,
                                 description="schema for loading one or more specific datasets for the analysis")




def outlierElim(ids, data, cont=0.05):
    od = IsolationForest(contamination=cont, behaviour="new")
    outlierIds = []
    for x in data:
        darr = data[x]
        f_outliers = od.fit_predict(darr)
        drop_o = np.nonzero(np.where(f_outliers==-1, 1, 0))[0]
        outlierIds.append(ids[drop_o])
    common = np.hstack(outlierIds)
    u, count_o = np.unique(common, return_counts=True)
    outlier = u[count_o>3]
    print(outlier)
    _, _, outlier_ind = np.intersect1d(outlier, ids, return_indices=True)
    np.savetxt(output_fld +'ids_outlier.csv', ids[outlier_ind], delimiter=",", fmt='%12.5f')
    ids = np.delete(ids, outlier_ind)
    np.savetxt(output_fld + 'ids_outlierDropped.csv', ids, delimiter=",", fmt='%12.5f')
    for x in data:
        data[x] = np.delete(data[x], outlier_ind, axis=0)
        np.savetxt(output_fld + x + '_outlierDropped.csv', data[x], delimiter=",", fmt='%12.5f')
    return ids, data


    
def equal_ar_size(array1, array2, label, i):
    r1, s1 = array1.shape
    r2, s2 = array2.shape
    if s1 > s2:
       array1 = signal.resample(array1, s2, axis=1)
       
    elif s2 > s1:
       array2 = signal.resample(array2, s1, axis=1)


    np.savetxt(output_fld + label + str(i) +'a1.csv', array1,delimiter=",", fmt='%12.5f')
    np.savetxt(output_fld + label + str(i) +'a2.csv', array2, delimiter=",", fmt='%12.5f')
    np.savetxt(output_fld + label + str(i) +'a1mean.csv', np.vstack((np.mean(array1, axis=0),np.std(array1,axis=0))),delimiter=",", fmt='%12.5f')
    np.savetxt(output_fld + label + str(i) +'a2mean.csv', np.vstack((np.mean(array2, axis=0),np.std(array2,axis=0))), delimiter=",", fmt='%12.5f')
    return array1, array2 
    

def normalize_ds(array1, norm_type):
    if norm_type == 1:
        #Scale to mean waveform
        scaler = preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(array1)
        #array1 = preprocessing.scale(array1, axis=1)
    elif norm_type == 2:
        #Scale by z score to pop mean
        array1 = preprocessing.scale(array1, axis=1)
    elif norm_type == 3:
        #manually Scale to mean waveform
        mean_wave = np.mean(array1, axis=0)
        mean_std = np.std(array1, axis=0)
        array1 = (array1 - mean_wave) / mean_std
    return array1



def main(params_file, output_dir, output_code, datasets, norm_type, labels_file, **kwargs):

    # Load data from each dataset
    data_objects = []
    specimen_ids_list = []
    imp = SimpleImputer(missing_values=0, strategy='mean', copy=False,)
    
    for ds in datasets:
        if len(ds["limit_to_cortical_layers"]) == 0:
            limit_to_cortical_layers = None
        else:
            limit_to_cortical_layers = ds["limit_to_cortical_layers"]

        data_for_spca, specimen_ids = ld.load_h5_data(h5_fv_file=ds["fv_h5_file"],
                                            metadata_file=ds["metadata_file"],
                                            dendrite_type=ds["dendrite_type"],
                                            need_structure=not ds["allow_missing_structure"],
                                            include_dend_type_null=ds["allow_missing_dendrite"],
                                            limit_to_cortical_layers=limit_to_cortical_layers,
                                            id_file=ds["id_file"],
                                            params_file=params_file)
        for l, m in data_for_spca.items():
            if type(m) == np.ndarray:
                nu_m = np.nan_to_num(m)
                p = np.nonzero(nu_m[:,:])[1]
                p = max(p)
                nu_m = nu_m[:,:p]
                print(l)
                print(p)
                nu_m = imp.fit_transform(nu_m)
                data_for_spca[l] = normalize_ds(nu_m, norm_type)
                
        data_objects.append(data_for_spca)
        specimen_ids_list.append(specimen_ids)
    specimen_ids = np.hstack(specimen_ids_list)
    
    data_for_spca = {}
    for i, do in enumerate(data_objects):
        for k in do:
            if k not in data_for_spca:
                data_for_spca[k] = do[k]
            else:
                data_for_spca[k], do[k] = equal_ar_size(data_for_spca[k], do[k], k, i)
                data_for_spca[k] = np.vstack([data_for_spca[k], do[k]])
    
    ##Outlier Elim? 
    #specimen_ids, data_for_spca = outlierElim(specimen_ids, data_for_spca)


    first_key = list(data_for_spca.keys())[0]
    if len(specimen_ids) != data_for_spca[first_key].shape[0]:
        logging.error("Mismatch of specimen id dimension ({:d}) and data dimension ({:d})".format(len(specimen_ids), data_for_spca[first_key].shape[0]))
    labels = pd.read_csv(labels_file, index_col=0)
    uni_labels = np.unique(labels.values)
    ids_list = labels.index.values
    
    if np.array_equal(ids_list, specimen_ids):
        print("Same Ids loaded... Proceeding")
        for p in data_for_spca:
            labels_means = pd.DataFrame()
            arr_data = data_for_spca[p]
            for x in uni_labels:
                indx = np.where(labels['0']==x)[0]
                mean = pd.Series(data=np.mean(arr_data[indx], axis=0), name=('Cluster ' + str(x) + ' mean'))
                std = pd.Series(data=np.std(arr_data[indx], axis=0), name=('Cluster ' + str(x) + ' std'))
                n = pd.Series(data=np.std(arr_data[indx], axis=0), name=('Cluster ' + str(x) + ' std'))
                labels_means = labels_means.append(mean, ignore_index=True)
                labels_means = labels_means.append(std, ignore_index=True)
                
            labels_means.to_csv(output_fld + p +'_cluster_mean.csv')
    logging.info("Proceeding with %d cells", len(specimen_ids))




   
    logging.info("Done.")


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=AnalysisParameters)
    main(**module.args)
