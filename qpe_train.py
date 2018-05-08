from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
# import google3
import six
from six.moves import xrange
import tensorflow as tf
import qpe_model
from proto.relation_pb2 import Relation
from tensorflow_fold.util import proto_tools
from google.protobuf import json_format

tf.flags.DEFINE_string(
    'train_data_path', '',
    'TF Record file containing the training dataset of expressions.')
tf.flags.DEFINE_integer(
    'batch_size', 10, 'How many samples to read per batch.')
tf.flags.DEFINE_integer(
    'embedding_length', 5,
    'How long to make the expression embedding vectors.')
tf.flags.DEFINE_integer(
    'max_steps', 1000000,
    'The maximum number of batches to run the trainer for.')

# Replication flags:
tf.flags.DEFINE_string('logdir', '/tmp/qpe/logs',
                       'Directory in which to write event logs.')
tf.flags.DEFINE_string('master', '',
                       'Tensorflow master to use.')
tf.flags.DEFINE_integer('task', 0,
                        'Task ID of the replica running the training.')
tf.flags.DEFINE_integer('ps_tasks', 0,
                        'Number of PS tasks in the job.')
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
    train_iterator = iterate_over_tf_record_protos(
        FLAGS.train_data_path, Relation)

    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):

            # Build the graph.
            classifier = qpe_model.QueryPerformanceEstimatorModel(FLAGS.embedding_length)
            loss = classifier.loss
            accuracy = classifier.accuracy
            train_op = classifier.train_op
            global_step = classifier.global_step

            # Set up the supervisor.
            supervisor = tf.train.Supervisor(logdir=FLAGS.logdir, is_chief=(FLAGS.task == 0), save_summaries_secs=10, save_model_secs=30)
            sess = supervisor.PrepareSession(FLAGS.master)

            # Run the trainer.
            for _ in xrange(FLAGS.max_steps):
                batch = [next(train_iterator) for _ in xrange(FLAGS.batch_size)]
                fdict = classifier.build_feed_dict(batch)

                _, step, loss_v, accuracy_v = sess.run(
                [train_op, global_step, loss, accuracy], feed_dict=fdict)
                print('step=%d: loss=%f accuracy=%f' % (step, loss_v, accuracy_v))
                emit_values(supervisor, sess, step, {'Batch Loss': loss_v, 'Batch Accuracy': accuracy_v})

if __name__ == '__main__':
    tf.app.run()
