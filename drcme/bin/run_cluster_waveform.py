import numpy as np
import pandas as pd
import drcme.spca_fit as sf
import drcme.load_data as ld
import argschema as ags
import joblib
import logging
import os
import json
from itertools import permutations
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy import signal
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn import tree
from sklearn import multiclass



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
    spca_file = ags.fields.InputFile()
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
        normalize = preprocessing.Normalizer(copy=False)
        scaler = preprocessing.StandardScaler(copy=False)
        np.nan_to_num(array1, copy=False)
        normalize.fit_transform(array1)
        #scaler.fit_transform(array1)
    elif norm_type == 4:
        #Scale by min max within sample
        array1 = preprocessing.minmax_scale(array1, (-1,1), axis=1, copy=False)
    return array1



def main(params_file, output_dir, output_code, datasets, norm_type, labels_file, spca_file, **kwargs):

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
    df_s = pd.read_csv(spca_file, index_col=0)

    first_key = list(data_for_spca.keys())[0]
    if len(specimen_ids) != data_for_spca[first_key].shape[0]:
        logging.error("Mismatch of specimen id dimension ({:d}) and data dimension ({:d})".format(len(specimen_ids), data_for_spca[first_key].shape[0]))
    labels = pd.read_csv(labels_file, index_col=0)
    uni_labels = np.unique(labels.values)
    ids_list = labels.index.values
    
    if labels.shape[0] == ids_list.shape[0]:
        print("Same Ids loaded... Proceeding")
        logging.info("Proceeding with %d cells", len(specimen_ids))
        for p in data_for_spca:
            labels_means = pd.DataFrame()
            arr_data = data_for_spca[p]
            for x in uni_labels:
                indx = np.where(labels['0']==x)[0]
                row, col = arr_data[indx].shape
                n_co = np.full(col, row)
                mean = pd.Series(data=np.mean(arr_data[indx], axis=0), name=('Cluster ' + str(x) + ' mean'))
                std = pd.Series(data=np.std(arr_data[indx], axis=0), name=('Cluster ' + str(x) + ' std'))
                n = pd.Series(data=n_co, name=('Cluster ' + str(x) + ' n'))
                labels_means = labels_means.append(mean, ignore_index=True)
                labels_means = labels_means.append(std, ignore_index=True)
                labels_means = labels_means.append(n, ignore_index=True)
            labels_means.to_csv(output_fld + p +'_cluster_mean.csv')
        
            
        train_df, test_df, labels_2, _ = train_test_split(df_s, labels)

        

        rf = RandomForestClassifier(n_estimators=500, oob_score=True,
                                         random_state=0)
        #per = multiclass.OneVsOneClassifier(RandomForestClassifier(n_estimators=500, oob_score=True,
                                        # random_state=0), n_jobs=-1).fit(train_df.values, labels.to_numpy().flatten())
        rf.fit(train_df.values, labels_2.to_numpy().flatten())
        logging.info("OOB score: {:f}".format(rf.oob_score_))
        pred_labels = rf.predict(test_df.values)
        feat_import = rf.feature_importances_
        print(rf.oob_score_)
        logging.debug("Saving results")
        pd.DataFrame(pred_labels, index=test_df.index.values).to_csv('rf_predictions.csv')
        pd.DataFrame(feat_import).to_csv('rf_feat_importance.csv')

    
    feat_import_by_label = np.hstack((0, np.full(feat_import.shape[0], np.nan)))
    for i in permutations(uni_labels, 2):
        indx_1 = np.where((labels['0']==i[0]))[0]
        indx_2 = np.where((labels['0']==i[1]))[0]
        indx = np.hstack((indx_1,indx_2))
        if indx.shape[0] >= 100:
            print(indx.shape[0])
            df_s_temp = df_s.iloc[indx]
            labels_s_temp = labels.iloc[indx]
            train_df, test_df, labels_2,_ = train_test_split(df_s_temp, labels_s_temp)

            

            rf = RandomForestClassifier(n_estimators=500, oob_score=True,
                                             random_state=0)
            #per = multiclass.OneVsOneClassifier(RandomForestClassifier(n_estimators=500, oob_score=True,
                                            # random_state=0), n_jobs=-1).fit(train_df.values, labels.to_numpy().flatten())
            rf.fit(train_df.values, labels_2.to_numpy().flatten())
            logging.info("OOB score: {:f}".format(rf.oob_score_))
            pred_labels = rf.predict(test_df.values)
            feat_import = rf.feature_importances_
            print(str(i)+ ' ' + str(rf.oob_score_))
            logging.debug("Saving results")
            feat_import_by_label = np.vstack((feat_import_by_label,np.hstack((str(i),np.ravel(feat_import)))))
            del rf    
    pd.DataFrame(feat_import_by_label).to_csv(output_fld + 'label_rf_feat_importance.csv')

   
    logging.info("Done.")


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=AnalysisParameters)
    main(**module.args)
