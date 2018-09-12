from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import tf_parameter_mgr

IMAGE_SIZE = 256

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1548
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 398


def read_zitie(filename_queue):

  class ZITIERecord(object):
    pass
  result = ZITIERecord()

  label_bytes = 1 
  result.height = IMAGE_SIZE
  result.width = IMAGE_SIZE
  result.depth = 1

  reader = tf.TFRecordReader()
  result.key, value = reader.read(filename_queue)
  features = tf.parse_single_example(value, features={
                                      'label': tf.FixedLenFeature([], tf.int64),
                                      'image_raw' : tf.FixedLenFeature([], tf.string)
                                     })
  result.label = tf.cast(features['label'], tf.int32)
  result.label = tf.expand_dims(result.label, 0)
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image = tf.reshape(image, [result.height, result.width, result.depth])
  # image = tf.expand_dims(image, -1)
  result.uint8image = image
  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])

def inputs(eval_data, data_dir, batch_size):
  if not eval_data:
    filenames = tf_parameter_mgr.getTrainData()
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = tf_parameter_mgr.getTestData()
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_zitie(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  # resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(reshaped_image)
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)
