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

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures 
   
    # model_input = tf.reshape(model_input, [-1,8,16])
    # gate_activations = tf.layers.conv1d(
    #     model_input,
    #     filters=1931 * (num_mixtures + 1),
    #     kernel_size=4,
    #     strides=4,
    #     padding='valid',
    #     activation=None
    #     )
    # gate_activations = tf.reshape(gate_activations, [3862 * (num_mixtures + 1), ])

    # expert_activations = tf.layers.conv1d(
    #     model_input,
    #     filters=1931 * (num_mixtures),
    #     kernel_size=4,
    #     strides=4,
    #     padding='valid',
    #     activation=None
    #     )
    # expert_activations = tf.reshape(expert_activations, [3862 * (num_mixtures), ]) 


    gate_activations = slim.fully_connected(
        model_input,                                  # B, 128
        vocab_size * (num_mixtures + 1),              # 3862 * (2 + 1) = (11586,)
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,                                  # B, 128
        vocab_size * num_mixtures,                    # 3862 * 2 = (7724,)
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")
    
    gating_distribution = tf.nn.softmax(tf.reshape(          
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1) # (3862, 3) 
    expert_distribution = tf.nn.sigmoid(tf.reshape(            
        expert_activations, 
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures           # (3862, 2)

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size]) # 
    return {"predictions": final_probabilities}

class NewMoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input_1,
                   model_input_2,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    #num_mixtures = num_mixtures or FLAGS.moe_num_mixtures 
   
    # model_input = tf.reshape(model_input, [-1,8,16])
    # gate_activations = tf.layers.conv1d(
    #     model_input,
    #     filters=1931 * (num_mixtures + 1),
    #     kernel_size=4,
    #     strides=4,
    #     padding='valid',
    #     activation=None
    #     )
    # gate_activations = tf.reshape(gate_activations, [3862 * (num_mixtures + 1), ])

    # expert_activations = tf.layers.conv1d(
    #     model_input,
    #     filters=1931 * (num_mixtures),
    #     kernel_size=4,
    #     strides=4,
    #     padding='valid',
    #     activation=None
    #     )
    # expert_activations = tf.reshape(expert_activations, [3862 * (num_mixtures), ]) 


    gate_activations_1 = slim.fully_connected(
        model_input_1,                                  # B, 128
        vocab_size,              # 3862 * (2 + 1) = (11586,)
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates_1")
    expert_activations_1 = slim.fully_connected(
        model_input_1,                                  # B, 128
        vocab_size,                    # 3862 * 2 = (7724,)
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts_1")
    
    gate_activations_2 = slim.fully_connected(
        model_input_2,                                  # B, 128
        vocab_size,              # 3862 * (2 + 1) = (11586,)
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates_2")
    expert_activations_2 = slim.fully_connected(
        model_input_2,                                  # B, 128
        vocab_size,                    # 3862 * 2 = (7724,)
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts_2")


    gate_activations = tf.keras.layers.concatenate([gate_activations_1, gate_activations_2])
    expert_activations = tf.keras.layers.concatenate([expert_activations_1, expert_activations_2])

    gating_distribution = tf.nn.softmax(tf.reshape(          
        gate_activations,
        [-1, 2]))  
    expert_distribution = tf.nn.sigmoid(tf.reshape(            
        expert_activations, 
        [-1, 2]))  

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size]) # 
    return {"predictions": final_probabilities}
