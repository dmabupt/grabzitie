from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import numpy as np

from six.moves import urllib
import tensorflow as tf
import tf_parameter_mgr

import zitie_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
batch_size=tf_parameter_mgr.getTrainBatchSize()

tf.app.flags.DEFINE_string('data_dir', '/tmp/zitie_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")


# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = zitie_input.IMAGE_SIZE
NUM_CLASSES = zitie_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = zitie_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = zitie_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = tf_parameter_mgr.getLearningRateDecay()  # Learning rate decay factor.
INITIAL_LEARNING_RATE = tf_parameter_mgr.getBaseLearningRate()       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  #with tf.device('/cpu:0'):
  #  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  #  var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def inputs(eval_data):
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'zitie-batches-bin')
  images, labels = zitie_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inference(images):
  # images is (1, 256, 256, 1)
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[7, 7, 1, 64], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)  # (1, 64, 64, 64)
  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')  # (1, 32, 32, 64)
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)   # (1, 32, 32, 128)
  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2') # (1, 16, 16, 128)
  
  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 256], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(pre_activation, name=scope.name)   # (1, 16, 16, 256)
  # norm3
  norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
  # pool3
  pool3 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3') # (1, 8, 8, 256)
  
  # local3
  with tf.variable_scope('local3') as scope:
    n_dim = np.prod(pool2.get_shape().as_list()[1:])
    reshape = tf.reshape(pool2, [-1, n_dim]) # (1, 2304)
    
    dim = reshape.get_shape()[1].value  # 2304
    weights = _variable_with_weight_decay('weights', shape=[dim, 1024],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[1024, 128],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [128, NUM_CLASSES],
                                          stddev=1/128.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
  return softmax_linear

def loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
  return total_loss
  
def accuracy(logits, labels):
  top_k_op = tf.nn.in_top_k(logits, labels, 1)
  correct = np.sum(top_k_op)
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
  return accuracy

def _add_loss_summaries(total_loss):
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  return loss_averages_op

def _train(total_loss, global_step, return_grad = False):
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  #tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf_parameter_mgr.getOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  if return_grad: return apply_gradient_op, grads
  return train_op
