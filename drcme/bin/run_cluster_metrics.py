import pandas as pd
import argschema as ags
import numpy as np
from sklearn import metrics
from sklearn import mixture
from scipy import stats
import matplotlib.pyplot as mp


### Script (in-part) includes modified functions from: 
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause




class MetricsParameters(ags.ArgSchema):
    ground_truth = ags.fields.InputFile()
    labels_file = ags.fields.InputFile()
    spca_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()
    cluster_drop = ags.fields.List(ags.fields.Int())


def main(ground_truth, labels_file, output_file, spca_file, cluster_drop, **kwargs):
    results = pd.DataFrame(columns=["name", "Base Score", "Random permutations", "KDE % above threshold", "KDE p value"])
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    ## Load Ground truth labels
    df_1 = pd.read_csv(ground_truth, index_col=0)
    df_2 = pd.read_csv(labels_file, index_col=0)
    ## Discard Everything with no ground truth
    df_s = pd.read_csv(spca_file, index_col=0)
    df_full = pd.concat([df_1, df_2], axis=1)
    drop_o = np.nonzero(np.where(df_full['line_num']==-1, 1, 0))[0]
    ## Also drop those from the predicted labels
    
    #Then drop the clusters desired
    for i in cluster_drop:
        drop_o = np.append(drop_o, np.nonzero(np.where(df_full['0']==i, 1, 0))[0])
    df_full = df_full.drop(df_full.index[drop_o], axis=0)
    df_full.to_csv('merge labelsoutput.csv')
    y_true = df_full['line_num'].to_numpy().flatten() 
    y_pred = df_full['0'].to_numpy().flatten()
    
    ## Compute threshold values for this dataset based on https://www.jneurosci.org/content/33/39/15454.full#sec-25##
    n_samples = len(y_true)
    n_clusters_range = [len(np.unique(y_pred))]
    n_runs = 1000
    for i, score_func in enumerate(score_funcs):
            results.loc[i,['name']] = score_f_label[i]
            T_score = score_func(y_true, y_pred)
            results.loc[i, ["Base Score"]] = T_score

            scores = uniform_labelings_scores(score_func, n_samples, y_pred, n_clusters_range, fixed_n_classes=y_true).flatten()
            results.loc[i,["Random permutations"]] = np.mean(scores)
            
            scores = fit_random_GMM(score_func, n_samples, y_true, df_s, n_clusters_range, n_runs, drop_o).flatten()
            above_threshold = np.nonzero(np.where(scores >= T_score, 1, 0))[0]
            results.loc[i,["KDE % above threshold"]] = len(above_threshold) / len(scores)
            hist = stats.gaussian_kde(scores, bw_method='scott')
            p_TR = hist.integrate_box_1d(T_score, scores.max())
            results.loc[i,["KDE p value"]] = p_TR



    print(results)
    results.to_csv(output_file)



def ami_score(U, V):
    return metrics.adjusted_mutual_info_score(U, V, average_method='arithmetic')

   


score_funcs = [
    metrics.adjusted_rand_score,
    metrics.v_measure_score,
    ami_score,
    metrics.homogeneity_score,
    metrics.completeness_score,
]

score_f_label = ['ARI', 'V_measure', 'AMI', 'Homegenity', 'Completeness']

def fit_random_GMM(score_func, n_samples, y_true, df_s, n_clusters_range,
                             n_runs=1000, drop_o=[]):
    """ Fit random intialized GMM to the DATA, then compare to the ground truth
    """
    
    spca_samples = df_s.to_numpy()
    scores = np.zeros((len(n_clusters_range), n_runs))
    gmm = mixture.GaussianMixture(n_components=n_clusters_range[0], covariance_type='diag', max_iter=20000, init_params='kmeans')
    labels_a = y_true

    for i, k in enumerate(n_clusters_range):
        for j in range(n_runs):
            labels_b = gmm.fit_predict(spca_samples)
            labels_b = np.delete(labels_b, drop_o)
            scores[i, j] = score_func(labels_a, labels_b)
    return scores


def uniform_labelings_scores(score_func, n_samples, y_pred, n_clusters_range, fixed_n_classes=None,
                             n_runs=1000):
    """Compute score for 1 random uniform cluster labelings.

    Both random labelings have the same number of clusters for each value
    possible value in ``n_clusters_range``.

    When fixed_n_classes is not None the first labeling is considered a ground
    truth class assignment with fixed number of classes.
    """
    random_labels = np.random.RandomState().randint
    random_permutation = np.random.RandomState().choice
    scores = np.zeros((len(n_clusters_range), n_runs))

    if fixed_n_classes is not None:
        if isinstance(fixed_n_classes, int):
            labels_a = random_labels(low=0, high=fixed_n_classes, size=n_samples)   
        else:
            labels_a = fixed_n_classes
    for i, k in enumerate(n_clusters_range):
        for j in range(n_runs):
            if fixed_n_classes is None:
                labels_a = random_labels(low=0, high=k, size=n_samples)
            labels_b = random_permutation(y_pred, size=n_samples)
            scores[i, j] = score_func(labels_a, labels_b)
    return scores



if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=MetricsParameters)
    main(**module.args)
