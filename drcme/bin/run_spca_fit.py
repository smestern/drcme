import numpy as np
import pandas as pd
import drcme.spca_fit as sf
import drcme.load_data as ld
import argschema as ags
import joblib
import logging
import os
import json
from ipfx.feature_vectors import _subsample_average
from sklearn.impute import KNNImputer
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
        array1 = preprocessing.scale(array1, axis=1)
        scaler = preprocessing.StandardScaler(csopy=False)
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

    return array1



def main(params_file, output_dir, output_code, datasets, norm_type, **kwargs):

    # Load data from each dataset
    data_objects = []
    specimen_ids_list = []
    imp = KNNImputer(copy=False)
    
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
                nu_m = m
                p = np.nonzero(nu_m[:,:])[1]
                p = max(p)
                
                print(l)
                print(p)
                data_for_spca[l] = nu_m
                
        data_objects.append(data_for_spca)
        specimen_ids_list.append(specimen_ids)

    data_for_spca = {}
    for i, do in enumerate(data_objects):
        for k in do:
            if k not in data_for_spca:
                do[k] = normalize_ds(do[k], norm_type)
                data_for_spca[k] = do[k]
            else:
                data_for_spca[k], do[k] = equal_ar_size(data_for_spca[k], do[k], k, i)
                
                do[k] = normalize_ds(do[k], norm_type)
                 
                data_for_spca[k] = np.vstack([data_for_spca[k], do[k]])
            np.savetxt(output_fld + k + str(i) +'.csv', do[k], delimiter=",", fmt='%12.5f')
            np.savetxt(output_fld + k + str(i) +'mean.csv', np.vstack((np.nanmean(do[k], axis=0),np.nanstd(do[k],axis=0))),delimiter=",", fmt='%12.5f')
    specimen_ids = np.hstack(specimen_ids_list)
    ### Now run through again and impute missing:
    for l in data_for_spca:
        nu_m = data_for_spca[l]
        nu_m = imp.fit_transform(nu_m)
        data_for_spca[l] = nu_m
    ##Outlier Elim? 
    #specimen_ids, data_for_spca = outlierElim(specimen_ids, data_for_spca)


    first_key = list(data_for_spca.keys())[0]
    if len(specimen_ids) != data_for_spca[first_key].shape[0]:
        logging.error("Mismatch of specimen id dimension ({:d}) and data dimension ({:d})".format(len(specimen_ids), data_for_spca[first_key].shape[0]))

    



    logging.info("Proceeding with %d cells", len(specimen_ids))




    # Load parameters
    spca_zht_params, _ = ld.define_spca_parameters(filename=params_file)

    # Run sPCA
    subset_for_spca = sf.select_data_subset(data_for_spca, spca_zht_params)
    spca_results = sf.spca_on_all_data(subset_for_spca, spca_zht_params)
    combo, component_record = sf.consolidate_spca(spca_results)

    logging.info("Saving results...")
    joblib.dump(spca_results, os.path.join(output_dir, "spca_loadings_{:s}.pkl".format(output_code)))
    combo_df = pd.DataFrame(combo, index=specimen_ids)
    combo_df.to_csv(os.path.join(output_dir, "sparse_pca_components_{:s}.csv".format(output_code)))
    with open(os.path.join(output_dir, "spca_components_used_{:s}.json".format(output_code)), "w") as f:
        json.dump(component_record, f, indent=4)
    logging.info("Done.")


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=AnalysisParameters)
    main(**module.args)
