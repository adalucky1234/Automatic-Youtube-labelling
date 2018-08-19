# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils
import numpy as np


import tensorflow.contrib.slim as slim
from tensorflow import flags
from tensorflow import logging

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_string("video_level_classifier_model_new", "NewMoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 128, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")
flags.DEFINE_integer("conv_layers", 2, "Number of Conv 1D layers.")
flags.DEFINE_integer("resnet_blocks", 3, "Number of resnet blocks.")

class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators

    output = slim.fully_connected(
        avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}

class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.get_variable("cluster_weights",
      [feature_size, cluster_size],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    tf.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
        [cluster_size],
        initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.get_variable("hidden1_weights",
      [cluster_size, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)



class NewSqueezeNetModel(models.BaseModel):
  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    def fire_module(self, x,sp,e11p,e33p):
        s = tf.layers.conv1d(x,sp,1,1,"same") 
        s = tf.nn.relu(s)
        
        e11 = tf.layers.conv1d(s,e11p,1,1,"same") 
        e11 = tf.nn.relu(e11)
    
        e33 = tf.layers.conv1d(s,e33p,3,1,"same") 
        e33 = tf.nn.relu(e33)
        return tf.concat([e11,e33],-1)



    conv1 = tf.layers.conv1d(
        model_input,
        filters=64,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu,
        name="conv1d_%d" % 1)


    pool1 = tf.layers.max_pooling1d(
      conv1,
      pool_size=3,
      strides=2,
      padding='same',
      name="pool1d_%d" % 1)

    fire2 = fire_module(self, pool1,16,64,64)
    fire3 = fire_module(self, fire2,16,64,64)

    pool4 = tf.layers.max_pooling1d(
      fire3,
      pool_size=3,
      strides=2,
      padding='same',
      name="pool1d_%d" % 1)

    fire5 = fire_module(self, pool4,32,128,128)
    fire6 = fire_module(self, fire5,32,128,128)

    pool7 = tf.layers.max_pooling1d(
      fire6,
      pool_size=3,
      strides=2,
      padding='same',
      name="pool1d_%d" % 1)

    fire8 = fire_module(self, pool7,48,192,192)
    fire9 = fire_module(self, fire8,48,192,192)
    fire10 = fire_module(self, fire9,64,256,256)
    fire11 = fire_module(self, fire10,64,256,256)

    conv12 = tf.layers.conv1d(
        fire11,
        filters=100,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=tf.nn.relu)

    avgpool13 = tf.layers.average_pooling1d(
      conv12,
      pool_size=4,
      strides=4,
      padding='same')

    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0
    logging.info("num_frames shape: " + str(tf.shape(num_frames)))
    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, avgpool13,
                                       sequence_length=tf.div(num_frames, tf.constant(30)),
                                       dtype=tf.float32)



    # outputs, state = tf.nn.dynamic_rnn(stacked_lstm, fire11,
    #                                    sequence_length=tf.add(tf.div(num_frames, tf.constant(8)), tf.constant(1)),
    #                                    dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)



class LstmCustomModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """


    conv1 = tf.layers.conv1d(
        model_input,
        filters=32,
        kernel_size=5,
        strides=1,
        activation=tf.nn.relu,
        padding="same")


    pool1 = tf.layers.max_pooling1d(
      conv1,
      pool_size=5,
      strides=2,
      padding="same")

    conv2 = tf.layers.conv1d(
        pool1,
        filters=64,
        kernel_size=3,
        strides=1,
        activation=tf.nn.relu,
        padding="same")

    pool2 = tf.layers.max_pooling1d(
      conv2,
      pool_size=5,
      strides=2,
      padding="same")

    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    logging.info("**** LSTM HIDDEN UNITS **** " + str(lstm_size))
    logging.info("**** LSTM LAYERS **** " + str(number_of_layers))

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, pool2,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)


class ParallelModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    # branch 1
    conv1_1 = tf.layers.conv1d(
        model_input,
        filters=32,
        kernel_size=5,
        strides=2,
        activation=tf.nn.relu,
        padding="same")


    pool1_1 = tf.layers.max_pooling1d(
      conv1_1,
      pool_size=5,
      strides=2,
      padding="same")

    conv1_2 = tf.layers.conv1d(
        pool1_1,
        filters=64,
        kernel_size=5,
        strides=2,
        activation=tf.nn.relu,
        padding="same")

    pool1_2 = tf.layers.max_pooling1d(
      conv1_2,
      pool_size=5,
      strides=2,
      padding="same")

    # branch 2
    conv2_1 = tf.layers.conv1d(
      model_input,
      filters=32,
      kernel_size=3,
      strides=2,
      activation=tf.nn.relu,
      padding="same")


    pool2_1 = tf.layers.max_pooling1d(
      conv2_1,
      pool_size=5,
      strides=2,
      padding="same")

    conv2_2 = tf.layers.conv1d(
        pool2_1,
        filters=64,
        kernel_size=3,
        strides=2,
        activation=tf.nn.relu,
        padding="same")

    pool2_2 = tf.layers.max_pooling1d(
      conv2_2,
      pool_size=5,
      strides=2,
      padding="same")

    #branch 3

    conv3_1 = tf.layers.conv1d(
      model_input,
      filters=32,
      kernel_size=7,
      strides=2,
      activation=tf.nn.relu,
      padding="same")


    pool3_1 = tf.layers.max_pooling1d(
      conv2_1,
      pool_size=5,
      strides=2,
      padding="same")

    conv3_2 = tf.layers.conv1d(
        pool2_1,
        filters=64,
        kernel_size=7,
        strides=2,
        activation=tf.nn.relu,
        padding="same")

    pool3_2 = tf.layers.max_pooling1d(
      conv2_2,
      pool_size=5,
      strides=2,
      padding="same")


    #branch 4

    conv4_1 = tf.layers.conv1d(
      model_input,
      filters=32,
      kernel_size=7,
      strides=2,
      activation=tf.nn.relu,
      padding="same")


    pool4_1 = tf.layers.max_pooling1d(
      conv2_1,
      pool_size=5,
      strides=2,
      padding="same")

    conv3_2 = tf.layers.conv1d(
        pool2_1,
        filters=64,
        kernel_size=7,
        strides=2,
        activation=tf.nn.relu,
        padding="same")

    pool3_2 = tf.layers.max_pooling1d(
      conv2_2,
      pool_size=5,
      strides=2,
      padding="same")


    concat3 = tf.keras.layers.concatenate([pool1_2, pool2_2, pool3_2])

    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers


    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, concat3,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)



class ImageModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    reshape0 = tf.reshape(model_input, [-1, 300, 384, 3])

    conv1 = tf.layers.conv2d(
        reshape0,
        filters=32,
        kernel_size=5,
        strides=1,
        activation=tf.nn.relu,
        padding="same")


    pool1 = tf.layers.max_pooling2d(
      conv1,
      pool_size=5,
      strides=2,
      padding="same")
    conv2 = tf.layers.conv2d(
        pool1,
        filters=64,
        kernel_size=3,
        strides=1,
        activation=tf.nn.relu,
        padding="same")

    pool2 = tf.layers.max_pooling2d(
      conv2,
      pool_size=5,
      strides=2,
      padding="same")

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=pool2,
        vocab_size=vocab_size,
        **unused_params)



class ParallelModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    # branch 1
    conv1_1 = tf.layers.conv1d(
        model_input,
        filters=32,
        kernel_size=5,
        strides=2,
        activation=tf.nn.relu,
        padding="same")


    pool1_1 = tf.layers.max_pooling1d(
      conv1_1,
      pool_size=5,
      strides=2,
      padding="same")

    conv1_2 = tf.layers.conv1d(
        pool1_1,
        filters=64,
        kernel_size=5,
        strides=2,
        activation=tf.nn.relu,
        padding="same")

    pool1_2 = tf.layers.max_pooling1d(
      conv1_2,
      pool_size=5,
      strides=2,
      padding="same")

    # branch 2
    conv2_1 = tf.layers.conv1d(
      model_input,
      filters=32,
      kernel_size=3,
      strides=2,
      activation=tf.nn.relu,
      padding="same")


    pool2_1 = tf.layers.max_pooling1d(
      conv2_1,
      pool_size=5,
      strides=2,
      padding="same")

    conv2_2 = tf.layers.conv1d(
        pool2_1,
        filters=64,
        kernel_size=3,
        strides=2,
        activation=tf.nn.relu,
        padding="same")

    pool2_2 = tf.layers.max_pooling1d(
      conv2_2,
      pool_size=5,
      strides=2,
      padding="same")

    #branch 3

    conv3_1 = tf.layers.conv1d(
      model_input,
      filters=32,
      kernel_size=7,
      strides=2,
      activation=tf.nn.relu,
      padding="same")


    pool3_1 = tf.layers.max_pooling1d(
      conv2_1,
      pool_size=5,
      strides=2,
      padding="same")

    conv3_2 = tf.layers.conv1d(
        pool2_1,
        filters=64,
        kernel_size=7,
        strides=2,
        activation=tf.nn.relu,
        padding="same")

    pool3_2 = tf.layers.max_pooling1d(
      conv2_2,
      pool_size=5,
      strides=2,
      padding="same")


    #branch 4

    conv4_1 = tf.layers.conv1d(
      model_input,
      filters=32,
      kernel_size=7,
      strides=2,
      activation=tf.nn.relu,
      padding="same")


    pool4_1 = tf.layers.max_pooling1d(
      conv2_1,
      pool_size=5,
      strides=2,
      padding="same")

    conv3_2 = tf.layers.conv1d(
        pool2_1,
        filters=64,
        kernel_size=7,
        strides=2,
        activation=tf.nn.relu,
        padding="same")

    pool3_2 = tf.layers.max_pooling1d(
      conv2_2,
      pool_size=5,
      strides=2,
      padding="same")


    concat3 = tf.keras.layers.concatenate([pool1_2, pool2_2, pool3_2])

    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers


    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, concat3,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)



class InceptionModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    # branch 1
    conv1_1 = tf.layers.conv1d(
        model_input,
        filters=32,
        kernel_size=1,
        strides=1,
        activation=tf.nn.relu,
        padding="same")


    pool1_1 = tf.layers.max_pooling1d(
      conv1_1,
      pool_size=5,
      strides=2,
      padding="same")

    conv1_2 = tf.layers.conv1d(
        pool1_1,
        filters=64,
        kernel_size=1,
        strides=1,
        activation=tf.nn.relu,
        padding="same")

    pool1_2 = tf.layers.max_pooling1d(
      conv1_2,
      pool_size=5,
      strides=2,
      padding="same")


    # branch 2
    conv2_1 = tf.layers.conv1d(
      model_input,
      filters=16,
      kernel_size=1,
      strides=1,
      activation=tf.nn.relu,
      padding="same")

    # branch 2, group 1
    conv2_1_1 = tf.layers.conv1d(
      conv2_1,
      filters=32,
      kernel_size=3,
      strides=1,
      activation=tf.nn.relu,
      padding="same")

    pool2_1_1 = tf.layers.max_pooling1d(
      conv2_1_1,
      pool_size=5,
      strides=2,
      padding="same")

    conv2_1_2 = tf.layers.conv1d(
        pool2_1_1,
        filters=64,
        kernel_size=3,
        strides=1,
        activation=tf.nn.relu,
        padding="same")

    pool2_1_2 = tf.layers.max_pooling1d(
      conv2_1_2,
      pool_size=5,
      strides=2,
      padding="same")

     # branch 2, group 2
    conv2_2_1 = tf.layers.conv1d(
      conv2_1,
      filters=32,
      kernel_size=5,
      strides=1,
      activation=tf.nn.relu,
      padding="same")

    pool2_2_1 = tf.layers.max_pooling1d(
      conv2_2_1,
      pool_size=5,
      strides=2,
      padding="same")

    conv2_2_2 = tf.layers.conv1d(
        pool2_2_1,
        filters=64,
        kernel_size=5,
        strides=1,
        activation=tf.nn.relu,
        padding="same")

    pool2_2_2 = tf.layers.max_pooling1d(
      conv2_2_2,
      pool_size=5,
      strides=2,
      padding="same")  


    # branch 2, group 3
    conv2_3_1 = tf.layers.conv1d(
      conv2_1,
      filters=32,
      kernel_size=7,
      strides=1,
      activation=tf.nn.relu,
      padding="same")

    pool2_3_1 = tf.layers.max_pooling1d(
      conv2_3_1,
      pool_size=5,
      strides=2,
      padding="same")

    conv2_3_2 = tf.layers.conv1d(
        pool2_3_1,
        filters=64,
        kernel_size=7,
        strides=1,
        activation=tf.nn.relu,
        padding="same")

    pool2_3_2 = tf.layers.max_pooling1d(
      conv2_3_2,
      pool_size=5,
      strides=2,
      padding="same")


    # branch 2, group 1
    conv2_4_1 = tf.layers.conv1d(
      conv2_1,
      filters=32,
      kernel_size=9,
      strides=1,
      activation=tf.nn.relu,
      padding="same")

    pool2_4_1 = tf.layers.max_pooling1d(
      conv2_4_1,
      pool_size=5,
      strides=2,
      padding="same")

    conv2_4_2 = tf.layers.conv1d(
        pool2_4_1,
        filters=64,
        kernel_size=9,
        strides=1,
        activation=tf.nn.relu,
        padding="same")

    pool2_4_2 = tf.layers.max_pooling1d(
      conv2_4_2,
      pool_size=5,
      strides=2,
      padding="same")


    concat3 = tf.keras.layers.concatenate([pool1_2, pool2_1_2, pool2_2_2, pool2_3_2, pool2_4_2])

    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers


    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, concat3,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)



class LstmMixtureModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):

    kernel_sizes = [5, 3, 5, 3, 5, 3]
    filters = [32, 64, 64, 128, 256, 512]

    conv_layers = FLAGS.conv_layers
    logging.info("**** CONV LAYERS **** " + str(conv_layers))


    conv1 = tf.layers.conv1d(
        model_input,
        filters=32,
        kernel_size=5,
        strides=1,
        activation=tf.nn.relu,
        padding="same")

    pool1 = tf.layers.max_pooling1d(
      conv1,
      pool_size=5,
      strides=2,
      padding="same")

    conv2 = tf.layers.conv1d(
        pool1,
        filters=64,
        kernel_size=3,
        strides=1,
        activation=tf.nn.relu,
        padding="same")

    pool2 = tf.layers.max_pooling1d(
      conv2,
      pool_size=5,
      strides=2,
      padding="same")


    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    logging.info("**** LSTM HIDDEN UNITS **** " + str(lstm_size))
    logging.info("**** LSTM LAYERS **** " + str(number_of_layers))

    with tf.variable_scope('lstm1'):
      stacked_lstm = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size, forget_bias=1.0)
                  for _ in range(number_of_layers)
                  ])

      loss = 0.0


      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, pool2,
                                         sequence_length=num_frames,
                                         dtype=tf.float32)



    # second LSTM

    conv2_1 = tf.layers.conv1d(
        model_input,
        filters=32,
        kernel_size=5,
        strides=1,
        activation=tf.nn.relu,
        padding="same")

    pool2_1 = tf.layers.max_pooling1d(
      conv2_1,
      pool_size=5,
      strides=2,
      padding="same")

    conv2_2 = tf.layers.conv1d(
        pool2_1,
        filters=64,
        kernel_size=5,
        strides=1,
        activation=tf.nn.relu,
        padding="same")

    pool2_2 = tf.layers.max_pooling1d(
      conv2_2,
      pool_size=5,
      strides=2,
      padding="same")


    lstm_size_2 = 256
    number_of_layers_2 = 1

    logging.info("**** LSTM HIDDEN UNITS 2 **** " + str(lstm_size_2))
    logging.info("**** LSTM LAYERS 2 **** " + str(number_of_layers_2))

    with tf.variable_scope('lstm2'):
      stacked_lstm_2 = tf.contrib.rnn.MultiRNNCell(
              [
                  tf.contrib.rnn.BasicLSTMCell(
                      lstm_size_2, forget_bias=1.0)
                  for _ in range(number_of_layers_2)
                  ])

      loss = 0.0

      outputs_2, state_2 = tf.nn.dynamic_rnn(stacked_lstm_2, pool2_2,
                                         sequence_length=num_frames,
                                         dtype=tf.float32)


    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model_new)

    return aggregated_model().create_model(
        model_input_1=state[-1].h,
        model_input_2=state_2[-1].h,
        vocab_size=vocab_size,
        **unused_params)



class LstmCustomNormModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):

    kernel_sizes = [5, 3, 5, 3, 5, 3]
    filters = [32, 64, 64, 128, 256, 512]

    conv_layers = FLAGS.conv_layers
    logging.info("**** CONV LAYERS **** " + str(conv_layers))


    conv1 = tf.layers.conv1d(
        model_input,
        filters=filters[0],
        kernel_size=kernel_sizes[0],
        strides=1,
        #activation=tf.nn.relu,
        padding="same")

    conv1 = tf.layers.batch_normalization(conv1, training=True) 
    conv1 = tf.nn.relu(conv1)

    pool1 = tf.layers.max_pooling1d(
      conv1,
      pool_size=5,
      strides=2,
      padding="same")

    conv2 = tf.layers.conv1d(
        pool1,
        filters=filters[1],
        kernel_size=kernel_sizes[1],
        strides=1,
        #activation=tf.nn.relu,
        padding="same")

    conv2 = tf.layers.batch_normalization(conv2, training=True) 
    conv2 = tf.nn.relu(conv2)

    pool2 = tf.layers.max_pooling1d(
      conv2,
      pool_size=5,
      strides=2,
      padding="same")


    inputLayer = pool2

    for i in range(2, conv_layers):
      

      conv = tf.layers.conv1d(
          inputLayer,
          filters=filters[i],
          kernel_size=kernel_sizes[i],
          strides=1,
          #activation=tf.nn.relu,
          padding="same")

      conv = tf.layers.batch_normalization(conv, training=True) 
      conv = tf.nn.relu(conv)

      pool = tf.layers.max_pooling1d(
        conv,
        pool_size=5,
        strides=2,
        padding="same")

      inputLayer = pool


    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    logging.info("**** LSTM HIDDEN UNITS **** " + str(lstm_size))
    logging.info("**** LSTM LAYERS **** " + str(number_of_layers))

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputLayer,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)



class LstmResnetNormModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):

    kernel_sizes = [3,3,3,3,3,3]
    filters = [256,256,256,256,256,256]

    resnet_blocks = FLAGS.resnet_blocks
    logging.info("**** resnet_blocks **** " + str(resnet_blocks))


    conv1 = tf.layers.conv1d(
        model_input,
        filters=256,
        kernel_size=1,
        strides=2,
        #activation=tf.nn.relu,
        padding="same")

    conv1 = tf.layers.batch_normalization(conv1, training=True) 
    conv1 = tf.nn.relu(conv1)

    pool1 = tf.layers.max_pooling1d(
      conv1,
      pool_size=5,
      strides=2,
      padding="same")

    inputLayer = pool1

    for i in range(0, len(filters), 2):

      conv = tf.layers.conv1d(
          inputLayer,
          filters=filters[i],
          kernel_size=kernel_sizes[i],
          strides=1,
          #activation=tf.nn.relu,
          padding="same")

      conv = tf.layers.batch_normalization(conv, training=True) 
      conv = tf.nn.relu(conv)

      conv = tf.layers.conv1d(
          conv,
          filters=filters[i],
          kernel_size=kernel_sizes[i],
          strides=1,
          #activation=tf.nn.relu,
          padding="same")

      conv = tf.keras.layers.Add()([conv, inputLayer])
      
      conv = tf.layers.batch_normalization(conv, training=True) 
      conv = tf.nn.relu(conv)

      inputLayer = conv


    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    logging.info("**** LSTM HIDDEN UNITS **** " + str(lstm_size))
    logging.info("**** LSTM LAYERS **** " + str(number_of_layers))

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputLayer,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)



class LstmCustomNormModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):

    kernel_sizes = [5, 3, 5, 3, 5, 3]
    filters = [32, 64, 64, 128, 256, 512]

    conv_layers = FLAGS.conv_layers
    logging.info("**** CONV LAYERS **** " + str(conv_layers))


    conv1 = tf.layers.conv1d(
        model_input,
        filters=filters[0],
        kernel_size=kernel_sizes[0],
        strides=1,
        #activation=tf.nn.relu,
        padding="same")

    conv1 = tf.layers.batch_normalization(conv1, training=True) 
    conv1 = tf.nn.relu(conv1)

    pool1 = tf.layers.max_pooling1d(
      conv1,
      pool_size=5,
      strides=2,
      padding="same")

    conv2 = tf.layers.conv1d(
        pool1,
        filters=filters[1],
        kernel_size=kernel_sizes[1],
        strides=1,
        #activation=tf.nn.relu,
        padding="same")

    conv2 = tf.layers.batch_normalization(conv2, training=True) 
    conv2 = tf.nn.relu(conv2)

    pool2 = tf.layers.max_pooling1d(
      conv2,
      pool_size=5,
      strides=2,
      padding="same")


    inputLayer = pool2

    for i in range(2, conv_layers):
      

      conv = tf.layers.conv1d(
          inputLayer,
          filters=filters[i],
          kernel_size=kernel_sizes[i],
          strides=1,
          #activation=tf.nn.relu,
          padding="same")

      conv = tf.layers.batch_normalization(conv, training=True) 
      conv = tf.nn.relu(conv)

      pool = tf.layers.max_pooling1d(
        conv,
        pool_size=5,
        strides=2,
        padding="same")

      inputLayer = pool


    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    logging.info("**** LSTM HIDDEN UNITS **** " + str(lstm_size))
    logging.info("**** LSTM LAYERS **** " + str(number_of_layers))

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputLayer,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)



class LstmResnetNormModel2(models.BaseModel):
  def resblock(self, inputLayer, filters, kernel_sizes):

    for i in range(0, len(filters), 2):

      conv = tf.layers.conv1d(
          inputLayer,
          filters=filters[i],
          kernel_size=kernel_sizes[i],
          strides=1,
          #activation=tf.nn.relu,
          padding="same")

      conv = tf.layers.batch_normalization(conv, training=True) 
      conv = tf.nn.relu(conv)

      conv = tf.layers.conv1d(
          conv,
          filters=filters[i],
          kernel_size=kernel_sizes[i],
          strides=1,
          #activation=tf.nn.relu,
          padding="same")

      conv = tf.keras.layers.Add()([conv, inputLayer])
      
      conv = tf.layers.batch_normalization(conv, training=True) 
      conv = tf.nn.relu(conv)
      inputLayer = conv

    return inputLayer

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):

    kernel_sizes = [3,3,3,3,3,3]
    filters = [256,256,256,256,256,256]


    resnet_blocks = FLAGS.resnet_blocks
    logging.info("**** resnet_blocks **** " + str(resnet_blocks))


    conv1 = tf.layers.conv1d(
        model_input,
        filters=256,
        kernel_size=5,
        strides=2,
        #activation=tf.nn.relu,
        padding="same")

    conv1 = tf.layers.batch_normalization(conv1, training=True) 
    conv1 = tf.nn.relu(conv1)

    pool1 = tf.layers.max_pooling1d(
      conv1,
      pool_size=5,
      strides=2,
      padding="same")

    inputLayer = self.resblock(pool1, filters, kernel_sizes)

    # add another resnet block

    residual = tf.layers.conv1d(
          inputLayer,
          filters=512,
          kernel_size=1,
          strides=2,
          #activation=tf.nn.relu,
          padding="same")

    conv = tf.layers.conv1d(
          inputLayer,
          filters=512,
          kernel_size=1,
          strides=2,
          #activation=tf.nn.relu,
          padding="same")

    conv = tf.layers.batch_normalization(conv, training=True) 
    conv = tf.nn.relu(conv)

    conv = tf.layers.conv1d(
          inputLayer,
          filters=512,
          kernel_size=1,
          strides=2,
          #activation=tf.nn.relu,
          padding="same")  

    conv = tf.keras.layers.Add()([conv, residual])
    
    conv = tf.layers.batch_normalization(conv, training=True) 
    conv = tf.nn.relu(conv)

    filters = [512,512,512,512,512,512]

    inputLayer = self.resblock(conv, filters, kernel_sizes)

    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    logging.info("**** LSTM HIDDEN UNITS **** " + str(lstm_size))
    logging.info("**** LSTM LAYERS **** " + str(number_of_layers))

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputLayer,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)



class LstmResnetNormModel3(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):

    # kernel_sizes = [3,3,3,3,3,3]
    # filters = [256,256,256,256,256,256]

    # resnet_blocks = FLAGS.resnet_blocks
    # logging.info("**** resnet_blocks **** " + str(resnet_blocks))


    conv1 = tf.layers.conv1d(
        model_input,
        filters=1152,
        kernel_size=3,
        strides=2,
        #activation=tf.nn.relu,
        padding="same")

    conv1 = tf.layers.batch_normalization(conv1, training=True) 
    conv1 = tf.nn.relu(conv1)

    pool1 = tf.layers.max_pooling1d(
      conv1,
      pool_size=5,
      strides=2,
      padding="same")

    inputLayer = pool1

 
    conv = tf.layers.conv1d(
        inputLayer,
        filters=1152,
        kernel_size=3,
        strides=1,
        #activation=tf.nn.relu,
        padding="same")

    conv = tf.layers.batch_normalization(conv, training=True) 
    conv = tf.nn.relu(conv)

    conv = tf.layers.conv1d(
        conv,
        filters=1152,
        kernel_size=3,
        strides=1,
        #activation=tf.nn.relu,
        padding="same")

    conv = tf.keras.layers.Add()([conv, inputLayer])
    
    conv = tf.layers.batch_normalization(conv, training=True) 
    conv = tf.nn.relu(conv)

    inputLayer = conv


    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    logging.info("**** LSTM HIDDEN UNITS **** " + str(lstm_size))
    logging.info("**** LSTM LAYERS **** " + str(number_of_layers))

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputLayer,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)



class LstmResnetNormModel4(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):

    # kernel_sizes = [3,3,3,3,3,3]
    # filters = [256,256,256,256,256,256]

    # resnet_blocks = FLAGS.resnet_blocks
    # logging.info("**** resnet_blocks **** " + str(resnet_blocks))


    conv1 = tf.layers.conv1d(
        model_input,
        filters=2304,
        kernel_size=3,
        strides=2,
        #activation=tf.nn.relu,
        padding="same")

    conv1 = tf.layers.batch_normalization(conv1, training=True) 
    conv1 = tf.nn.relu(conv1)
    logging.info("conv1 shape " + str(conv1.get_shape()))

    pool1 = tf.layers.max_pooling1d(
      conv1,
      pool_size=5,
      strides=2,
      padding="same")

    inputLayer = pool1

    logging.info("pool1 shape " + str(pool1.get_shape()))
 
    conv = tf.layers.conv1d(
        inputLayer,
        filters=2304,
        kernel_size=3,
        strides=1,
        #activation=tf.nn.relu,
        padding="same")

    conv = tf.layers.batch_normalization(conv, training=True) 
    conv = tf.nn.relu(conv)

    logging.info("conv shape " + str(conv.get_shape()))

    conv = tf.layers.conv1d(
        conv,
        filters=2304,
        kernel_size=3,
        strides=1,
        #activation=tf.nn.relu,
        padding="same")

    conv = tf.keras.layers.Add()([conv, inputLayer])
    
    conv = tf.layers.batch_normalization(conv, training=True) 
    conv = tf.nn.relu(conv)

    inputLayer = conv

    logging.info("conv shape " + str(conv.get_shape()))


    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    logging.info("**** LSTM HIDDEN UNITS **** " + str(lstm_size))
    logging.info("**** LSTM LAYERS **** " + str(number_of_layers))

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputLayer,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)


class Resnet6BlowupLSTMModel(models.BaseModel):
  def resblock(self, inputLayer, filters, kernel_sizes):

    for i in range(0, len(filters), 2):

      conv = tf.layers.conv1d(
          inputLayer,
          filters=filters[i],
          kernel_size=kernel_sizes[i],
          strides=1,
          #activation=tf.nn.relu,
          padding="same")

      conv = tf.layers.batch_normalization(conv, training=True) 
      conv = tf.nn.relu(conv)

      conv = tf.layers.dropout(conv, rate=0.2)

      conv = tf.layers.conv1d(
          conv,
          filters=filters[i],
          kernel_size=kernel_sizes[i],
          strides=1,
          #activation=tf.nn.relu,
          padding="same")

      conv = tf.keras.layers.Add()([conv, inputLayer])
      
      conv = tf.layers.batch_normalization(conv, training=True) 
      conv = tf.nn.relu(conv)

      conv = tf.layers.dropout(conv, rate=0.2)

      inputLayer = conv

    return inputLayer

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):

    kernel_sizes = [3,3,3,3,3,3]
    filters = [4096,4096,4096,4096,4096,4096]


    resnet_blocks = FLAGS.resnet_blocks
    logging.info("**** resnet_blocks **** " + str(resnet_blocks))

    conv1 = tf.layers.conv1d(
        model_input,
        filters=4096,
        kernel_size=5,
        strides=2,
        #activation=tf.nn.relu,
        padding="same")

    conv1 = tf.layers.batch_normalization(conv1, training=True) 
    conv1 = tf.nn.relu(conv1)

    logging.info("conv1 shape" + str(conv1.get_shape()))

    pool1 = tf.layers.max_pooling1d(
      conv1,
      pool_size=2,
      strides=2,
      padding="same")

    pool1 = tf.layers.dropout(pool1, rate=0.2)

    logging.info("pool1 shape" + str(pool1.get_shape()))

    inputLayer = self.resblock(pool1, filters, kernel_sizes)

    logging.info("RESNET OUPUT shape" + str(inputLayer.get_shape()))
    

    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    logging.info("**** LSTM HIDDEN UNITS **** " + str(lstm_size))
    logging.info("**** LSTM LAYERS **** " + str(number_of_layers))

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputLayer,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=tf.layers.dropout(state[-1].h, rate=0.2),
        vocab_size=vocab_size,
        l2_penalty=1e-6,
        **unused_params)

