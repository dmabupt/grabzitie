#! /usr/bin/env python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import glob
import numpy as np
from six.moves import xrange
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the training.')


tf.app.flags.DEFINE_string('train_dir', '/tmp/zitie_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
#tf.app.flags.DEFINE_integer('max_steps', 1000000,
#tf.app.flags.DEFINE_integer('max_steps', 1000,
#                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
                           
import zitie
# import tf_parameter_mgr
# import monitor_cb

max_steps=1000
test_interval=15

#tf.logging.set_verbosity(tf.logging.INFO)

def setup_distribute():
    global FLAGS
    worker_hosts = []
    ps_hosts = []
    spec = {}
    if FLAGS.worker_hosts is not None and FLAGS.worker_hosts != '':
        worker_hosts = FLAGS.worker_hosts.split(',')
        spec.update({'worker': worker_hosts})

    if FLAGS.ps_hosts is not None and FLAGS.ps_hosts != '':
        ps_hosts = FLAGS.ps_hosts.split(',')
        spec.update({'ps': ps_hosts})

    if len(worker_hosts) > 0:
        print('Cluster spec: ', spec)
        cluster = tf.train.ClusterSpec(spec)

        # Create and start a server for the local task.
        server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)
        if FLAGS.job_name == "ps":
            server.join()
    else:
        cluster = None
        server = tf.train.Server.create_local_server()
        # enforce a task_id for single node mode
        FLAGS.task_id = 0

    return cluster, server

def train():
    cluster, server = setup_distribute()

    is_chief = (FLAGS.task_id == 0)
    
    if is_chief:
      log_dir = os.path.join(FLAGS.train_dir, 'log')
      # monitor = monitor_cb.CMonitor(log_dir, tf_parameter_mgr.getTestInterval(), tf_parameter_mgr.getMaxSteps())
      # summaryWriter = tf.summary.FileWriter(log_dir)
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_id,
        cluster=cluster)):
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        is_training = tf.placeholder(dtype=bool,shape=())
        i_train, l_train = zitie.inputs(False)
        i_test, l_test = zitie.inputs(True)
        images, labels = tf.cond(is_training, lambda:(i_train, l_train), lambda:(i_test, l_test))
        #images, labels = zitie.distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        #logits = zitie.inference(images
        logits = zitie.inference(images)
        # Calculate loss.
        loss = zitie.loss(logits, labels)
        accuracy = zitie.accuracy(logits, labels)
        train_op = zitie.train(loss, global_step)
        
        if is_chief:
          graph = tf.get_default_graph()
          for layer in ['conv1', 'conv2', 'local3', 'local4']:
            monitor.SummaryHist("weight", graph.get_tensor_by_name(layer+'/weights:0'), layer)
            monitor.SummaryHist("bias", graph.get_tensor_by_name(layer+'/biases:0'), layer)
            monitor.SummaryHist("activation", graph.get_tensor_by_name(layer+'/'+layer+':0'), layer)
            monitor.SummaryNorm2("weight", graph.get_tensor_by_name(layer+'/weights:0'), layer)
          monitor.SummaryGradient("weight", loss)
          monitor.SummaryGWRatio()
          monitor.SummaryScalar("train loss", loss)
          monitor.SummaryScalar("train accuracy", accuracy)
          monitor.SummaryScalar("test loss", loss)
          monitor.SummaryScalar("test accuracy", accuracy)
          train_summaries = tf.summary.merge_all()
          test_summaries = tf.summary.merge_all()

    class _LoggerHook(tf.train.SessionRunHook):
        """Logs loss and runtime."""
    
        def begin(self):
                self._next_trigger_step = test_interval
                self._trigger = False
    
        def before_run(self, run_context):
            args = {'global_step': global_step}
            if self._trigger:
                self._trigger = False
                args['summary'] = train_summaries
            return tf.train.SessionRunArgs(args)  # Asks for loss value.
    
        def after_run(self, run_context, run_values):
            gs = run_values.results['global_step']
            if gs >= self._next_trigger_step:
                self._trigger = True
                self._next_trigger_step += test_interval

            # summary = run_values.results.get('summary', None)
            # if summary is not None:
                # summaryWriter.add_summary(summary, gs)
                # summary = run_context.session.run(test_summaries, feed_dict = {is_training:False})
                # summaryWriter.add_summary(summary, gs)

    hooks = [tf.train.StopAtStepHook(last_step=tf_parameter_mgr.getMaxSteps()),
               tf.train.NanTensorHook(loss)]
    if is_chief: hooks.append(_LoggerHook())
    with tf.train.MonitoredTrainingSession(
        master=server.target,
        is_chief=is_chief,
        checkpoint_dir=FLAGS.train_dir,
        hooks=hooks,
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      steps = 0
      while not mon_sess.should_stop():
          mon_sess.run(train_op, feed_dict = {is_training:True})
          steps += 1
          if steps % 100 == 0: print('%d stpes executed on worker %d.' % (steps, FLAGS.task_id))
      print('%d stpes executed on worker %d.' % (steps, FLAGS.task_id))
      if is_chief:
          summaryWriter.flush()

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()

