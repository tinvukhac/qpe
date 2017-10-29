from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# import google3
from google.protobuf import json_format
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from qpe_proto.relation_pb2 import Relation

tf.flags.DEFINE_string('output_path', '/tmp/qpe/dataset',
                       'Where to write the TFRecord file with the expressions.')
tf.flags.DEFINE_integer('max_depth', 5,
                        'Maximum expression depth.')
tf.flags.DEFINE_integer('num_samples', 1000,
                        'How many samples to put into the table.')
FLAGS = tf.flags.FLAGS


def random_relation(max_depth):
    def build(relation, max_depth):
        if max_depth == 0 or random.uniform(0, 1) < 1.0 / 3.0:
            relation.rows = random.choice(range(50, 100))
            relation.width = random.choice(range(10, 20))
            relation.op = Relation.NONE
        else:
            if max_depth < 0:
                return
            op = Relation.NONE
            while op == Relation.NONE:
                op = random.choice(Relation.OpCode.values())
            relation.op = op
            number_of_relations = random.choice(range(1, 5))
            for i in range(number_of_relations):
                relation.relations.extend([Relation()])
                build(relation.relations[i], max_depth - 1)

    while True:
        relation = Relation()
        build(relation, max_depth)
        if relation.op != Relation.NONE:
            break

    return relation


def make_random_relation():
    max_depth = random.choice(range(1, 5))
    relation = random_relation(max_depth)
    relation.result = random.choice(range(3))


def main():
    record_output = tf.python_io.TFRecordWriter(FLAGS.output_path)
    for _ in xrange(FLAGS.num_samples):
        record_output.write(make_random_relation())
    record_output.close()


if __name__ == "__main__":
    main()
