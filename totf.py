import math
import os
import sys
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

labels = { "csl":0, "htj":1, "mf":2, "oyx":3, "sgt":4, "wxz":5, "wxaz":6, "xbq":7, "yzq":8, "zsj":9 }

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  
def get_trainging_data(training_dir, validation_dir):
  with tf.Session() as sess:
    with tf.python_io.TFRecordWriter('train.tfrecords') as record_writer:
      for dirname in os.listdir(training_dir):
        dirpath = os.path.join(training_dir, dirname)  # zitie/yzq
        print(dirpath)
        if os.path.isdir(dirpath):
          label = labels[dirname] # label = yzq
          for filename in os.listdir(dirpath): 
            if filename.endswith('jpg'):
              filepath = os.path.join(dirpath, filename)  # filepath = zitie/yzq/0001.png
              print(filepath)
              image_raw_data = tf.gfile.FastGFile(filepath, 'rb').read()
              # if(filename.endswith('jpg')):
              img_data = tf.image.decode_jpeg(image_raw_data)
              # img_data = tf.image.resize_images(img_data, [256, 256])
              #img_data = tf.image.convert_image_dtype(img_data, dtype = tf.float32)
              img_data = tf.image.rgb_to_grayscale(img_data)
              #img_data = tf.squeeze(img_data)
              # img_data = tf.where(img_data>10, 255*tf.ones_like(img_data), tf.zeros_like(img_data))
              print(img_data.eval().shape)
              # plt.figure(1)
              # plt.imshow(tf.squeeze(img_data).eval())
              # plt.title(label)
              # plt.show()
              example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(img_data.eval().tobytes()),
                'label': _int64_feature(label)
              }))
              record_writer.write(example.SerializeToString())
    with tf.python_io.TFRecordWriter('test.tfrecords') as record_writer:
      for dirname in os.listdir(validation_dir):
        dirpath = os.path.join(validation_dir, dirname)  # zitie/yzq
        print(dirpath)
        if os.path.isdir(dirpath):
          label = labels[dirname] # label = yzq
          for filename in os.listdir(dirpath): 
            if filename.endswith('jpg'):
              filepath = os.path.join(dirpath, filename)  # filepath = zitie/yzq/0001.png
              print(filepath)
              image_raw_data = tf.gfile.FastGFile(filepath, 'rb').read()
              img_data = tf.image.decode_jpeg(image_raw_data)
              img_data = tf.image.rgb_to_grayscale(img_data)
              print(img_data.eval().shape)
              example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(img_data.eval().tobytes()),
                'label': _int64_feature(label)
              }))
              record_writer.write(example.SerializeToString())
            
def main():
    get_trainging_data('zitie', 'zitie-val')

if __name__ == "__main__":
    main()