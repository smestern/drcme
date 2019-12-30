import pandas as pd
import argschema as ags
import numpy as np
from sklearn import metrics


class MetricsParameters(ags.ArgSchema):
    ground_truth = ags.fields.InputFile()
    labels_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()



def main(ground_truth, labels_file, output_file, **kwargs):
    results = pd.DataFrame()
    ## Load Ground truth labels
    df_1 = pd.read_csv(ground_truth, index_col=0)
    ## Discard Everything with no ground truth
    drop_o = np.nonzero(np.where(df_1==-1, 1, 0))[0]
    df_1 = df_1.drop(df_1.index[drop_o], axis=0)
    ## Also drop those from the predicted labels
    df_2 = pd.read_csv(labels_file, index_col=0)
    df_2 = df_2.drop(df_2.index[drop_o], axis=0)
    y_true, y_pred = df_1.to_numpy().flatten(), df_2.to_numpy().flatten()
    results['Adjusted Rand Index'] = [metrics.adjusted_rand_score(y_true, y_pred)]

    results['Adjusted Mutual Information'] = [metrics.adjusted_mutual_info_score(y_true, y_pred)]

    results['V Measure'] = [metrics.v_measure_score(y_true, y_pred, beta=1.8)]

    results['Homogeneity'] = [metrics.homogeneity_score(y_true, y_pred)]
    results['completeness'] = [metrics.completeness_score(y_true, y_pred)]

    results['fowlkes mallows score'] = [metrics.v_measure_score(y_true, y_pred)]

    print(results)
    results.to_csv(output_file)



if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=MetricsParameters)
    main(**module.args)
