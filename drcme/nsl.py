import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble
import sklearn.preprocessing as pre
import logging
import itertools
import pickle
import time
import scipy
import neural_structured_learning as nsl

from absl import app
from absl import flags
from absl import logging
import numpy as np
import six
import tensorflow as tf

class HParams(object):
  """Hyperparameters used for training."""
  def __init__(self):
    ### dataset parameters
    self.num_classes = 7
    self.max_seq_length = 700
    self.vocab_size = 10000
    ### neural graph learning parameters
    self.distance_type = nsl.configs.DistanceType.L2
    self.graph_regularization_multiplier = 0.1
    self.num_neighbors = 3
    ### model architecture
    self.num_classes = 10
    self.conv_filters = [32, 64, 64]
    self.kernel_size = 3
    self.pool_size = 2
    self.num_fc_units = [64]
    self.num_embedding_dims = 16
    self.num_lstm_dims = 64
    self.dropout_rate = 0.2
    ### training parameters
    self.train_epochs = 15
    self.batch_size = 10

    ### eval parameters
    self.eval_steps = None  # All instances in the test set are evaluated.


NBR_FEATURE_PREFIX = 'NL_nbr_'
NBR_WEIGHT_SUFFIX = '_weight'

"""
This code is replicated from Google's Neural Structural Learning Package & Juypter notebook. Slightly modified to
Fit the needs of this project. 
This code adapts the following files:
--build_graph.py
--pack_nbrs.py
--https://github.com/tensorflow/neural-structured-learning/blob/master/g3doc/tutorials/graph_keras_lstm_imdb.ipynb

As such, this portion of the code is subject to the copyright and license it was
originally released under. This is outlined here:

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""



_MIN_NORM = np.float64(1e-6)


def _read_spca_examples(spca, id_feature_name, embedding_feature_name):
  """Reads and returns the embeddings stored in the Examples in `filename`.
  Args:
    filenames: A list of names of TFRecord files containing `tf.train.Example`
      objects.
    id_feature_name: Name of the feature that identifies the Example's ID.
    embedding_feature_name: Name of the feature that identifies the Example's
        embedding.
  Returns:
    A dict mapping each instance ID to its L2-normalized embedding, represented
    by a 1-D numpy.ndarray. The ID is expected to be contained in the singleton
    bytes_list feature named by 'id_feature_name', and the embedding is
    expected to be contained in the float_list feature named by
    'embedding_feature_name'.
  """
 
  def l2_normalize(v):
    """Returns the L2-norm of the vector `v`.
    Args:
      v: A 1-D vector (either a list or numpy array).
    Returns:
      The L2-normalized version of `v`. The result will have an L2-norm of 1.0.
    """
    l2_norm = np.linalg.norm(v)
    return v / max(l2_norm, _MIN_NORM)

  embeddings = {}
  
  
  for ex_id in spca.index.values:
    embedding_list = spca.loc[ex_id].values
    embeddings[ex_id] = l2_normalize(embedding_list)
  
  return embeddings


def _write_edges(embeddings, threshold, f):
  """Writes relevant edges to `f` among pairs of the given `embeddings`.
  This function considers all distinct pairs of nodes in `embeddings`,
  computes the dot product between all such pairs, and writes any edge to `f`
  for which the similarity is at least the given `threshold`.
  Args:
    embeddings: A `dict`: node_id -> embedding.
    threshold: A `float` representing an inclusive lower-bound on the cosine
        similarity for an edge to be added.
    f: A file object to which all edges are written in TSV format. The caller is
        responsible for opening and closing this file.
  Returns:
    The number of bi-direction edges written to the file.
  """
  start_time = time.time()
  edge_cnt = 0
  all_combos = itertools.combinations(six.iteritems(embeddings), 2)
  for (i, emb_i), (j, emb_j) in all_combos:
    weight = np.dot(emb_i, emb_j)
    if weight >= threshold:
      f.write('%s\t%s\t%f\n' % (i, j, weight))
      f.write('%s\t%s\t%f\n' % (j, i, weight))
      edge_cnt += 1
      if (edge_cnt % 1000000) == 0:
        logging.info('Wrote %d edges in %.2f seconds....', edge_cnt,
                     (time.time() - start_time))

  return edge_cnt


def build_graph(spca,
                output_graph_path,
                similarity_threshold=0.8,
                id_feature_name='id',
                embedding_feature_name='embedding'):
  
  embeddings = _read_spca_examples(spca, id_feature_name,
                                       embedding_feature_name)
  start_time = time.time()
  logging.info('Building graph and writing edges to TSV file: %s',
               output_graph_path)
  with open(output_graph_path, 'w') as f:
    edge_cnt = _write_edges(embeddings, similarity_threshold, f)
    logging.info(
        'Wrote graph containing %d bi-directional edges (%.2f seconds).',
        edge_cnt, (time.time() - start_time))


def _main(argv):
  """Main function for invoking the `nsl.tools.build_graph` function."""
  flag = flags.FLAGS
  flag.showprefixforinfo = False
  if len(argv) < 3:
    raise app.UsageError(
        'Invalid number of arguments; expected 2 or more, got %d' %
        (len(argv) - 1))

  #build_graph(argv[1:-1], argv[-1], flag.similarity_threshold,
  #            flag.id_feature_name, flag.embedding_feature_name)

def _int64_feature(value):
  """Returns int64 tf.train.Feature."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value.tolist()))


def _bytes_feature(value):
  """Returns bytes tf.train.Feature."""
  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))


def _float_feature(value):
  """Returns float tf.train.Feature."""
  py_list = np.ravel(value).tolist()
  return tf.train.Feature(float_list=tf.train.FloatList(value=py_list))


def create_example(vector, label, record_id):
  """Create tf.Example containing the vector, label, and ID."""
  features = {
      'id': _bytes_feature(str(record_id)),
      'waves': _float_feature(np.asarray(vector)),
      'label': _int64_feature(np.asarray(label)),
  }
  return tf.train.Example(features=tf.train.Features(feature=features))



def build_base_model():
  """Builds a model according to the architecture defined in `hparams`."""
  inputs = tf.keras.Input(
      shape=(HPARAMS.max_seq_length,), dtype=tf.float32, name='waves')

  x = inputs
  x = tf.keras.layers.Reshape((HPARAMS.max_seq_length,1), input_shape=(HPARAMS.max_seq_length,))(x)
  for i, num_filters in enumerate(HPARAMS.conv_filters):
    x = tf.keras.layers.Conv1D(
        num_filters, HPARAMS.kernel_size, activation='relu')(
            x)
    if i < len(HPARAMS.conv_filters) - 1:
      # max pooling between convolutional layers
      x = tf.keras.layers.MaxPooling1D(HPARAMS.pool_size)(x)
  x = tf.keras.layers.Flatten()(x)
  for num_units in HPARAMS.num_fc_units:
    x = tf.keras.layers.Dense(num_units, activation='relu')(x)
  pred = tf.keras.layers.Dense(HPARAMS.num_classes, activation='softmax')(x)
  model = tf.keras.Model(inputs=inputs, outputs=pred)
  return model



def graph_nsl(train_path, full_path, training_samples_count=10):
    print("### RUNNING GRAPH NETWORK ####")
    HPARAMS.max_seq_length = training_samples_count.shape[1]
    HPARAMS.vocab_size = 10
    train_dataset = make_dataset(train_path, True)
    pred_dataset = make_dataset(full_path)
    
    test_fraction = 0.3
    test_size = int(test_fraction *
                      int(training_samples_count.shape[0] // HPARAMS.batch_size))
    
    test_dataset = train_dataset.take(test_size)
    train_dataset = train_dataset.skip(test_size)
    validation_fraction = 0.2
    validation_size = int(validation_fraction *
                      int(training_samples_count.shape[0] // HPARAMS.batch_size))
    print('taking val: ' + str(validation_size) + ' test: ' + str(int(( 1 - validation_fraction) *
                      int(training_samples_count.shape[0]))))
    validation_dataset = train_dataset.take(validation_size)
    train_dataset = train_dataset.skip(validation_size)
    
   
    base_reg_model = build_base_model()
    
    graph_reg_config = nsl.configs.make_graph_reg_config(
    max_neighbors=HPARAMS.num_neighbors,
    multiplier=HPARAMS.graph_regularization_multiplier,
    distance_type=HPARAMS.distance_type,
    sum_over_axis=-1)
    graph_reg_model = nsl.keras.GraphRegularization(base_reg_model,
                                                graph_reg_config)
    graph_reg_model.compile(
        optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(base_reg_model.summary())
    
    graph_reg_history = graph_reg_model.fit(
    train_dataset, validation_data=validation_dataset,
    epochs=HPARAMS.train_epochs)
    graph_reg_model.evaluate(test_dataset)

    predict = graph_reg_model.predict(pred_dataset)

    return predict


def save_for_gam(data, labels, dataset='none'):
    sparse_data = scipy.sparse.csr_matrix(np.asarray(data))
    encoder = pre.OneHotEncoder()
    labels_oh = encoder.fit_transform(labels)



def pad_sequence(sequence, max_seq_length):
  """Pads the input sequence (a `tf.SparseTensor`) to `max_seq_length`."""
  pad_size = tf.maximum([0], max_seq_length - tf.shape(sequence)[0])
  padded = tf.concat(
      [sequence.values,
       tf.fill((pad_size), tf.cast(0, sequence.dtype))],
      axis=0)
  # The input sequence may be larger than max_seq_length. Truncate down if
  # necessary.
  return tf.slice(padded, [0], [max_seq_length])

def _parse_example(example_proto):
  
  feature_spec = {
      'waves': tf.io.VarLenFeature(tf.float32),
      'label': tf.io.FixedLenFeature((), tf.int64, default_value=-1),
  }
  
  for i in range(HPARAMS.num_neighbors):
    nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'waves')
    nbr_weight_key = '{}{}{}'.format(NBR_FEATURE_PREFIX, i, NBR_WEIGHT_SUFFIX)
    feature_spec[nbr_feature_key] = tf.io.VarLenFeature(tf.float32)

    # We assign a default value of 0.0 for the neighbor weight so that
    # graph regularization is done on samples based on their exact number
    # of neighbors. In other words, non-existent neighbors are discounted.
    feature_spec[nbr_weight_key] = tf.io.FixedLenFeature(
        [1], tf.float32, default_value=tf.constant([0.0]))

  features = tf.io.parse_single_example(example_proto, feature_spec)

  features['waves'] = tf.sparse.to_dense(features['waves'])
  for i in range(HPARAMS.num_neighbors):
    nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'waves')
    features[nbr_feature_key] = pad_sequence(features[nbr_feature_key],
                                             HPARAMS.max_seq_length)
    
    
  

 
  labels = features.pop('label')
  return features, labels

def make_dataset(file_path, training=False):
  """Creates a `tf.data.TFRecordDataset`.

  Args:
    file_path: Name of the file in the `.tfrecord` format containing
      `tf.train.Example` objects.
    training: Boolean indicating if we are in training mode.

  Returns:
    An instance of `tf.data.TFRecordDataset` containing the `tf.train.Example`
    objects.
  """
  dataset = tf.data.TFRecordDataset(file_path)
  if training:
    dataset = dataset.shuffle(10000)
  dataset = dataset.map(_parse_example)
  
  dataset = dataset.batch(HPARAMS.batch_size)
  return dataset

HPARAMS = HParams()

if __name__ == '__main__':
  flags.DEFINE_string(
      'id_feature_name', 'id',
      """Name of the singleton bytes_list feature in each input Example
      whose value is the Example's ID.""")
  flags.DEFINE_string(
      'embedding_feature_name', 'embedding',
      """Name of the float_list feature in each input Example
      whose value is the Example's (dense) embedding.""")
  flags.DEFINE_float(
      'similarity_threshold', 0.8,
      """Lower bound on the cosine similarity required for an edge
      to be created between two nodes.""")

  # Ensure TF 2.0 behavior even if TF 1.X is installed.
  tf.compat.v1.enable_v2_behavior()
  app.run(_main)