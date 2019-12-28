import numpy as np
import pandas as pd
import argschema as ags
from sklearn.externals import joblib
import drcme.load_data as ld
from drcme.spca_transform import *
import logging
from sklearn.impute import SimpleImputer

class DatasetParameters(ags.schemas.DefaultSchema):
    fv_h5_file = ags.fields.InputFile(description="HDF5 file with feature vectors")
    metadata_file = ags.fields.InputFile(description="Metadata file in CSV format", allow_none=True, default=None)
    dendrite_type = ags.fields.String(default="all", validate=lambda x: x in ["all", "spiny", "aspiny"])
    allow_missing_structure = ags.fields.Boolean(required=False, default=False)
    allow_missing_dendrite = ags.fields.Boolean(required=False, default=False)
    limit_to_cortical_layers = ags.fields.List(ags.fields.String, default=[], cli_as_single_argument=True)
    id_file = ags.fields.InputFile(description="Text file with IDs to use",
        required=False, allow_none=True, default=None)


class SpcaTransformParameters(ags.ArgSchema):
    orig_transform_file = ags.fields.InputFile(description="sPCA loadings file")
    orig_datasets = ags.fields.Nested(DatasetParameters,
        required=True,
        many=True,
        description="schema for loading one or more specific datasets for the analysis")
    new_datasets = ags.fields.Nested(DatasetParameters,
        required=True,
        many=True,
        description="schema for loading one or more specific datasets for the analysis")
    params_file = ags.fields.InputFile(default="/allen/aibs/mat/nathang/single-cell-ephys/dev/default_spca_params.json")
    output_file = ags.fields.OutputFile(description="CSV with transformed values")
    use_noise = ags.fields.Boolean(default=False)


def main(orig_transform_file, orig_datasets, new_datasets, params_file,
         output_file, use_noise, **kwargs):
    spca_zht_params, _ = ld.define_spca_parameters(params_file)

    spca_results = joblib.load(orig_transform_file)
    imp = SimpleImputer(missing_values=0, strategy='mean', copy=False,)
    # These arguments should be parameterized
    orig_data_objects = []
    orig_specimen_ids_list = []
    for ds in orig_datasets:
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
                data_for_spca[l] = nu_m
                
        orig_data_objects.append(data_for_spca)
        orig_specimen_ids_list.append(specimen_ids)
    orig_data_for_spca = {}
    for i, do in enumerate(orig_data_objects):
        for k in do:
            if k not in orig_data_for_spca:
                orig_data_for_spca[k] = do[k]
            else:
                orig_data_for_spca[k] = np.vstack([orig_data_for_spca[k], do[k]])
    orig_specimen_ids = np.hstack(orig_specimen_ids_list)
    logging.info("Original datasets had {:d} cells".format(len(orig_specimen_ids)))
    orig_mean, orig_std = orig_mean_and_std_for_zscore_h5(spca_results, orig_data_for_spca, spca_zht_params)

    new_data_objects = []
    new_specimen_ids_list = []
    for ds in new_datasets:
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
                data_for_spca[l] = nu_m
                
        new_data_objects.append(data_for_spca)
        new_specimen_ids_list.append(specimen_ids)
    data_for_spca = {}
    for i, do in enumerate(new_data_objects):
         for k in do:
            if k not in data_for_spca:
                data_for_spca[k] = do[k]
            else:
                data_for_spca[k] = np.vstack([data_for_spca[k], do[k]])
    new_ids = np.hstack(new_specimen_ids_list)
    logging.info("Applying transform to {:d} new cells".format(len(new_ids)))
    new_combo = spca_transform_new_data_h5(spca_results,
                                        data_for_spca,
                                        spca_zht_params,
                                        orig_mean, orig_std)
    new_combo_df = pd.DataFrame(new_combo, index=new_ids)
    new_combo_df.to_csv(output_file)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=SpcaTransformParameters)
    main(**module.args)