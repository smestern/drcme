import pandas as pd
import argschema as ags
import drcme.tsne as tsne
import sklearn.semi_supervised as sm
import numpy as np

class ComboTsneParameters(ags.ArgSchema):
    spca_file_1 = ags.fields.InputFile()
    labels_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()
    n_neighbours = ags.fields.Integer(default=600)
    gamma = ags.fields.Float(default=25)
    n_iter = ags.fields.Integer(default=20000)


def main(spca_file_1, labels_file, output_file,
         n_neighbours, gamma, n_iter, **kwargs):
    df_1 = pd.read_csv(spca_file_1, index_col=0).to_numpy()
    y_gt = pd.read_csv(labels_file, index_col=0).to_numpy().ravel()
    output_df = pd.DataFrame()
    prop = sm.LabelSpreading(kernel='knn', n_neighbors=n_neighbours, gamma=gamma, max_iter=n_iter, n_jobs=-1)
    prop.fit(df_1, y_gt)
    output = prop.transduction_.reshape(-1,1)  
    output_prob = prop.predict_proba(df_1)
    output_dist = prop.label_distributions_
    np.savetxt("label_prop.csv", np.hstack((output,output_prob,output_dist)), delimiter=',')


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=ComboTsneParameters)
    main(**module.args)
