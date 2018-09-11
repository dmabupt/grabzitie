import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
data_path = 'training.tfrecords'  # address to save the hdf5 file
with tf.Session() as sess:
  feature = {
    'image_raw': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64)
  }
  # Create a list of filenames and pass it to a queue
  filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
  # Define a reader and read the next record
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  # Decode the record read by the reader
  features = tf.parse_single_example(serialized_example, features=feature)
  # Convert the image data from string back to the numbers
  image = tf.decode_raw(features['image_raw'], tf.float32)
  print(image.shape)
  # Cast label data into int32
  label = tf.cast(features['label'], tf.int32)
  # Reshape image data into the original shape
  image = tf.reshape(image, [256, 256])
  print(image.shape)