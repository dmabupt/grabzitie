#! /usr/bin/env python2
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, argparse
import tensorflow as tf
import cifar10
from meta_writer import *

DEFAULT_CKPT_DIR = './train'
DEFAULT_WEIGHT_FILE = ''
DEFAULT_MODEL_FILE = 'mymodel.model'
DEFAULT_META_FILE = 'mymodel.meta'
DEFAULT_GRAPH_FILE = 'mymodel.graph'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('weights', DEFAULT_WEIGHT_FILE,
                           "Weight file to be loaded in order to validate, inference or continue train")
tf.app.flags.DEFINE_string('train_dir', DEFAULT_CKPT_DIR,
                           "checkpoint directory to resume previous train and/or snapshot current train, default to \"%s\"" % (DEFAULT_CKPT_DIR))
tf.app.flags.DEFINE_string('model_file', DEFAULT_MODEL_FILE,
                           "model file name to export, default to \"%s\"" % (DEFAULT_MODEL_FILE))
tf.app.flags.DEFINE_string('meta_file', DEFAULT_META_FILE,
                           "meta file name to export, default to \"%s\"" % (DEFAULT_META_FILE))
tf.app.flags.DEFINE_string('graph_file', DEFAULT_GRAPH_FILE,
                           "graph file name to export, default to \"%s\"" % (DEFAULT_GRAPH_FILE))


def main(unused_args):
    images, labels = cifar10.distorted_inputs()
    eimages, elabels = cifar10.inputs(True)
    
    
    global_step = tf.contrib.framework.get_or_create_global_step()
    logits = cifar10.inference(images)
    loss = cifar10.loss(logits, labels)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(top_k_op, "float"), name="accuracy")
    optApplyOp, grads_and_vars = cifar10.train(loss, global_step, True)
    
    tf.get_variable_scope().reuse_variables()
    elogits=cifar10.inference(eimages)
    eloss = cifar10.loss(elogits, elabels)
    etop_k_op = tf.nn.in_top_k(elogits, elabels, 1)
    eaccuracy = tf.reduce_mean(tf.cast(etop_k_op, "float"), name="eaccuracy")
    
    checkpoint_file = os.path.join(FLAGS.train_dir, "model.ckpt")
    restore_file = FLAGS.weights
    snapshot_interval = 100
    write_meta(tf,None,accuracy,loss, eaccuracy, eloss, optApplyOp, grads_and_vars,global_step,FLAGS.model_file, FLAGS.meta_file,FLAGS.graph_file,restore_file,checkpoint_file,snapshot_interval)

if __name__ == '__main__':
    tf.app.run()
    
