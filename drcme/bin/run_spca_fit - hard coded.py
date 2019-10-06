import numpy as np
import pandas as pd
import drcme.spca_fit as sf
import drcme.load_data as ld
import argschema as ags
import joblib
import logging
import os
import json


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
    datasets = ags.fields.Nested(DatasetParameters,
                                 required=True,
                                 many=True,
                                 description="schema for loading one or more specific datasets for the analysis")


def main(params_file, output_dir, output_code, datasets, **kwargs):

    # Load data from each dataset
    data_objects = []
    specimen_ids_list = []
    
    data_for_spca, specimen_ids = ld.load_h5_data("C:\\Users\\SMest\\source\\repos\\drcme\\drcme\\bin\\fv_test.h5",
                                            metadata_file=None,
                                            limit_to_cortical_layers=None,
                                            id_file="C:\\Users\\SMest\\source\\repos\\drcme\\drcme\\bin\\specids.txt",
                                            params_file="C:\\Users\\SMest\\source\\repos\\drcme\\drcme\\bin\\default_spca_params.json")
    data_objects.append(data_for_spca)
    specimen_ids_list.append(specimen_ids)

    data_for_spca = {}
    for i, do in enumerate(data_objects):
        for k in do:
            if k not in data_for_spca:
                data_for_spca[k] = do[k]
            else:
                data_for_spca[k] = np.vstack([data_for_spca[k], do[k]])
    specimen_ids = np.hstack(specimen_ids_list)

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
    main("C:\\Users\\SMest\\source\\repos\\drcme\\drcme\\bin\\default_spca_params.json", "C:\\Users\\SMest\\source\\repos\\drcme\\drcme\\bin\\output", "test",  1)
    

    ## --output_dir "C:\Users\SMest\source\repos\drcme\drcme\bin\output" --output_code "test" --input_json "C:\Users\SMest\source\repos\drcme\drcme\bin\dataset_params.json" --output_json "C:\Users\SMest\source\repos\drcme\drcme\bin"