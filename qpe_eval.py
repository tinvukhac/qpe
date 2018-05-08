"""Runs the evaluator for the query performance estimator test."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import os
# import google3
import six
from six.moves import xrange
import tensorflow as tf
from tensorflow_fold.util import proto_tools
import qpe_model
from proto.relation_pb2 import Relation

tf.flags.DEFINE_string(
    'validation_data_path',
    '',
    'TF Record containing the validation dataset of expressions.')
tf.flags.DEFINE_integer(
    'batch_size', 10, 'How many samples to read per batch.')
tf.flags.DEFINE_integer(
    'embedding_length', 5,
    'How long to make the expression embedding vectors.')
tf.flags.DEFINE_string(
    'eval_master', '',
    'Tensorflow master to use.')
tf.flags.DEFINE_string(
    'logdir', '/tmp/qpe/logs',
    'Directory where we read models and write event logs.')
tf.flags.DEFINE_integer(
    'eval_interval_secs', 10,
    'Time interval between eval runs. Zero to do a single eval then exit.')
FLAGS = tf.flags.FLAGS


# Find the root of the bazel repository.
def source_root():
    root = __file__
    for _ in xrange(5):
        root = os.path.dirname(root)
    return root

QPE_SOURCE_ROOT = source_root()
QPE_PROTO_FILE = ('proto/relation.proto')
QPE_EXPRESSION_PROTO = ('qpe.proto.Relation')


# Make sure serialized_message_to_tree can find the relation example proto:
proto_tools.map_proto_source_tree_path('', QPE_SOURCE_ROOT)
proto_tools.import_proto_file(QPE_PROTO_FILE)


def iterate_over_tf_record_protos(table_path, unused_message_type):
    while True:
        for v in tf.python_io.tf_record_iterator(table_path):
            yield proto_tools.serialized_message_to_tree(QPE_EXPRESSION_PROTO, v)


def emit_values(supervisor, session, step, values):
    summary = tf.Summary()
    for name, value in six.iteritems(values):
        summary_value = summary.value.add()
        summary_value.tag = name
        summary_value.simple_value = float(value)
    supervisor.summary_computed(session, summary, global_step=step)


def main(unused_argv):
    print('Reading validation table...')
    validation_iterator = iterate_over_tf_record_protos(
        FLAGS.validation_data_path, Relation)
    print('Done reading validation table...')

    with tf.Graph().as_default():
        # global_step = tf.Variable(0, name='global_step', trainable=False)
        classifier = qpe_model.QueryPerformanceEstimatorModel(FLAGS.embedding_length)
        global_step = classifier.global_step

        loss = classifier.loss
        accuracy = classifier.accuracy

        saver = tf.train.Saver()
        supervisor = tf.train.Supervisor(
            logdir=FLAGS.logdir,
            recovery_wait_secs=FLAGS.eval_interval_secs)
        sess = supervisor.PrepareSession(
            FLAGS.eval_master,
            wait_for_checkpoint=True,
            start_standard_services=False)

        validation_data = [next(validation_iterator) for _ in xrange(FLAGS.batch_size)]

        while not supervisor.ShouldStop():
            ckpt = tf.train.get_checkpoint_state(FLAGS.logdir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                continue
            step, validation_loss, validation_accuracy = sess.run(
                [global_step, loss, accuracy],
                feed_dict=classifier.build_feed_dict(validation_data))
            print('Step %d:  loss=%f accuracy=%f' % (
                step, validation_loss, validation_accuracy))
            emit_values(supervisor, sess, step,
                           {'Validation Loss': validation_loss,
                            'Validation Accuracy': validation_accuracy})
            if not FLAGS.eval_interval_secs: break
            time.sleep(FLAGS.eval_interval_secs)

        supervisor.Stop()


if __name__ == '__main__':
    tf.app.run()
