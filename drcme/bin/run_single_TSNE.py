import pandas as pd
import argschema as ags
import drcme.tsne as tsne
import matplotlib.pyplot as plt
import numpy as np

class ComboTsneParameters(ags.ArgSchema):
    spca_file_1 = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()
    n_components = ags.fields.Integer(default=2)
    perplexity = ags.fields.Float(default=25.)
    n_iter = ags.fields.Integer(default=20000)


def main(spca_file_1, output_file,
         n_components, perplexity, n_iter, **kwargs):
    df_1 = pd.read_csv(spca_file_1, index_col=0)
    row = int((len(df_1.index) / 2))
    df_2 = df_1.iloc[row:]
    df_1 = df_1.iloc[:row]
    full_df = pd.DataFrame()
    #learning_rate=10, early_exaggeration=500
    for per in np.arange(5, perplexity):
        combo_df = tsne.combined_tsne(df_1, df_2, n_components, per, n_iter)
        #combo_df.to_csv(output_file)
        combo_df.plot.scatter(x='x', y='y')
        plt.title(str(per))
        
        full_df['x' + str(per)] = combo_df['x'].values
        full_df['y' + str(per)] = combo_df['y'].values
    print('finished')
    
    full_df.to_csv(output_file)
    plt.show()


    


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=ComboTsneParameters)
    main(**module.args)
