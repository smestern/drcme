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

class DatasetParameters(ags.schemas.DefaultSchema):
    csv_file = ags.fields.InputFile(description="Metadata file in CSV format", allow_none=False, default=None)
    fv_h5_file = ags.fields.OutputFile(description="HDF5 file with feature vectors")
    
    


def main(csv_file, output):
    pd_data = pd.read_csv(csv_file)

    
    logging.info("Done.")


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=DatasetParameters)
    main(**module.args)
    

    