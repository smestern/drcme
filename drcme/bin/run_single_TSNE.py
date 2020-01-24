import pandas as pd
import argschema as ags
import drcme.tsne as tsne
import matplotlib.pyplot as plt


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
    #learning_rate=10, early_exaggeration=500
    combo_df = tsne.combined_tsne(df_1, df_2, n_components, perplexity, n_iter)
    combo_df.plot.scatter(x='x', y='y')
    combo_df.to_csv(output_file)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=ComboTsneParameters)
    main(**module.args)
