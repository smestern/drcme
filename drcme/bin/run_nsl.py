import numpy as np
import pandas as pd
import drcme.spca_fit as sf
import drcme.load_data as ld
import drcme.nsl as nsl_tools
import drcme.spca_pack_nbrs as spca_pack_nbrs
import argschema as ags
import joblib
import logging
import os
import json
import matplotlib.pyplot as plt
from drcme.gam.models import gcn
from drcme.gam.data import dataset
from drcme.gam.trainer import trainer_classification_gcn
from ipfx.feature_vectors import _subsample_average
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from scipy import signal
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import math
import neural_structured_learning as nsl
import tensorflow as tf
import cleanlab

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
    spca_file = ags.fields.InputFile()
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

def labelnoise(spca, labels, threshold=0.7):
    ##Fit with random forest
    lb_idx = np.where(labels.values != -1)[0]
    x = spca.values[lb_idx]
    lb = np.ravel(labels.values[lb_idx])
    
    rf = RandomForestClassifier(n_estimators=10, oob_score=True,
                                             random_state=0)
    rf.fit(x, lb)
    prob = np.amax(rf.predict_proba(x), axis=1)   
        

    prob_below = np.where(prob < threshold, -1, lb)


    labels.values[lb_idx] = prob_below.reshape(-1,1)
    return labels.values

    
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
        scaler = preprocessing.StandardScaler(copy=False)
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
    elif norm_type == 5:
        #Scale by min max within sample
        normalize = preprocessing.Normalizer(copy=False)

        baseline = np.mean(array1[:,:30], axis=1).reshape(-1,1)
        array1 = array1 - baseline
        array1 = preprocessing.minmax_scale(array1, (-1,1), axis=1, copy=False)
    return array1



def main(params_file, output_dir, output_code, datasets, norm_type, labels_file, spca_file, **kwargs):

    # Load data from each dataset
    data_objects = []
    specimen_ids_list = []
    imp = KNNImputer(copy=False)
    pad_len = 0
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
                if p > pad_len:
                    pad_len = p
                data_for_spca[l] = nu_m
                
        data_objects.append(data_for_spca)
        specimen_ids_list.append(specimen_ids)
    HPARAMS = HParams()

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
    labels = pd.read_csv(labels_file, index_col=0)
    
    df_s = pd.read_csv(spca_file, index_col=0)

    #labels['0'] = labelnoise(df_s, labels)
    train_ind = np.where(labels['0'] > -1)[0]
    pred_ind = np.where(labels['0'] == -1)[0]
    train_id = specimen_ids[train_ind]
    pred_id = specimen_ids[pred_ind]
    train_label = labels.iloc[train_ind]
    train_label = labels.iloc[train_ind]
    pred_label = labels.iloc[pred_ind]
    ### Now run through again and impute missing:
    train_data = {}
    pred_data = {}
    
    for l in data_for_spca:
        nu_m = data_for_spca[l]
        nu_m = imp.fit_transform(nu_m)
        if nu_m.shape[1] < pad_len:
            pad_wid = (pad_len - nu_m.shape[1]) + 1
            nu_m = np.hstack((nu_m, np.zeros((nu_m.shape[0], pad_wid))))
        train_data[l] = nu_m[train_ind]
        pred_data[l] = nu_m[pred_ind]
        data_for_spca[l] = nu_m
    ##Outlier Elim? 
    #specimen_ids, data_for_spca = outlierElim(specimen_ids, data_for_spca)
    ## Form our datasets for training
    HPARAMS.input_shape =  [pad_len + 1,1, len(data_for_spca)]
    full_data = np.hstack((data_for_spca[i] for i in sorted(data_for_spca.keys())))
    train_data = np.hstack((train_data[i] for i in sorted(train_data.keys())))
    pred_data = np.hstack((pred_data[i] for i in sorted(pred_data.keys())))    
    

    first_key = list(data_for_spca.keys())[0]
    if len(specimen_ids) != data_for_spca[first_key].shape[0]:
        logging.error("Mismatch of specimen id dimension ({:d}) and data dimension ({:d})".format(len(specimen_ids), data_for_spca[first_key].shape[0]))
   
       
    ## Write the Data to a record for use with 'graph params'
    writer = tf.io.TFRecordWriter(output_dir + 'train_data.tfr')
    for id, data, label in zip(train_id, train_data, train_label.values):
        example = nsl_tools.create_example(data,label,id)
        writer.write(example.SerializeToString())
    writer = tf.io.TFRecordWriter(output_dir + 'pred_data.tfr')
    for id, data, label in zip(specimen_ids, full_data, labels.values):
        example = nsl_tools.create_example(data,label,id)
        writer.write(example.SerializeToString())
    
    
    
    

    logging.info("Proceeding with %d cells", len(specimen_ids))
    
    base_model = tf.keras.Sequential([
    tf.keras.Input(train_data.shape[1], name='feature'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

 
    #Split into validation / test / train datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
      {'feature': train_data, 'label': np.ravel(train_label.values)}).shuffle(200)
    test_fraction = 0.3
    test_size = int(test_fraction *
                      int(train_data.shape[0]))
    
    test_dataset = train_dataset.take(test_size).batch(10)
    train_dataset = train_dataset.skip(test_size)
    validation_fraction = 0.2
    validation_size = int(validation_fraction *
                      int(train_data.shape[0]))
    print('taking val: ' + str(validation_size) + ' test: ' + str(int(( 1 - validation_fraction) *
                      int(train_data.shape[0]))))
    validation_dataset = train_dataset.take(validation_size).batch(10)
    train_dataset = train_dataset.skip(validation_size).batch(10)




    # Wrap the model with adversarial regularization. 
    adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.05) 
    adv_model = nsl.keras.AdversarialRegularization(base_model,
        adv_config=adv_config)
    # Compile, train, and evaluate. 
    full_labels = np.where(labels.values != -1, labels.values, (np.unique(labels.values)[-1] + 1))
    adv_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    history = adv_model.fit(train_dataset, validation_data=validation_dataset, epochs=5) 
    print('### FIT COMPLETE ### TESTING')
    adv_model.evaluate(test_dataset)
    pred_labels_prob = adv_model.predict({'feature': full_data, 'label':full_labels })


    pred_labels = np.argmax(pred_labels_prob, axis=1)
    logging.info("Saving results...")
    labels['0'] = pred_labels
    
    
    labels.to_csv(output_code + '_NSL_pred_adv_learn.csv')


    ####GRAPH NETWORK
    ##nsl_tools.save_for_gam(full_data, full_labels)

    nsl_tools.build_graph(df_s, output_dir + 'embed.tsv')
    spca_pack_nbrs.pack_nbrs(
       output_dir + '/train_data.tfr',
        output_dir + '/pred_data.tfr',
        output_dir + 'embed.tsv',
        output_dir + '/nsl_train_data.tfr',
    add_undirected_edges=True,
    max_nbrs=6)
    predictions = nsl_tools.graph_nsl(output_dir + '/nsl_train_data.tfr', output_dir + '/pred_data.tfr', full_data)
    pred_labels = np.argmax(predictions, axis=1)
    logging.info("Saving results...")
    labels['0'] = pred_labels
    
    
    
    labels.to_csv(output_code + '_NSL_pred_graph_learn.csv')
    logging.info("Done.")



class HParams(object):
  def __init__(self):
    self.input_shape = [28, 28, 1]
    self.num_classes = 6
    self.conv_filters = [32, 64, 64]
    self.kernel_size = (3, 3)
    self.pool_size = (2, 2)
    self.num_fc_units = [64]
    self.batch_size = 32
    self.epochs = 5
    self.adv_multiplier = 0.2
    self.adv_step_size = 0.2
    self.adv_grad_norm = 'infinity'

def build_base_model(hparams):
  """Builds a model according to the architecture defined in `hparams`."""
  inputs = tf.keras.Input(
      shape=hparams.input_shape, dtype=tf.float32)

  x = inputs
  for i, num_filters in enumerate(hparams.conv_filters):
    x = tf.keras.layers.Conv2D(
        num_filters, hparams.kernel_size, activation='relu', padding='same')(
            x)
    if i < len(hparams.conv_filters) - 1:
      # max pooling between convolutional layers
      x = tf.keras.layers.MaxPooling2D(hparams.pool_size, padding='same')(x)
  x = tf.keras.layers.Flatten()(x)
  for num_units in hparams.num_fc_units:
    x = tf.keras.layers.Dense(num_units, activation='relu')(x)
  pred = tf.keras.layers.Dense(hparams.num_classes, activation='softmax')(x)
  model = tf.keras.Model(inputs=inputs, outputs=pred)
  return model

if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=AnalysisParameters)
    main(**module.args)
