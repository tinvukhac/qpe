from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import random
from google.protobuf import json_format
from proto.relation_pb2 import Relation


tf.flags.DEFINE_string('output_path', '',
                       'Where to write the TFRecord file with the expressions.')
tf.flags.DEFINE_integer('max_depth', 5,
                        'Maximum operators tree depth.')
tf.flags.DEFINE_integer('num_samples', 1000,
                        'How many samples to put into the table.')
tf.flags.DEFINE_string('output_format', 'proto',
                       'Where to write the TFRecord file with the expressions.')
FLAGS = tf.flags.FLAGS

system1_weight_bias_dict = {"JOIN": {"weight": 1.5, "bias": 1.0},
                            "MAPJOIN": {"weight": 1.2, "bias": 1.0},
                            "EXTRACT": {"weight": 1.0, "bias": 1.0},
                            "FILTER": {"weight": 1.2, "bias": 1.0}}

system2_weight_bias_dict = {"JOIN": {"weight": 1.1, "bias": 1.0},
                            "MAPJOIN": {"weight": 1.5, "bias": 1.0},
                            "EXTRACT": {"weight": 1.1, "bias": 1.0},
                            "FILTER": {"weight": 1.0, "bias": 1.0}}

system3_weight_bias_dict = {"JOIN": {"weight": 1.2, "bias": 1.0},
                            "MAPJOIN": {"weight": 1.5, "bias": 1.0},
                            "EXTRACT": {"weight": 1.0, "bias": 1.0},
                            "FILTER": {"weight": 1.0, "bias": 1.0}}


def evaluate_relation(relation, weight_bias_dict):
    if relation.op == Relation.NONE:
        return relation.rows
    x = 0
    for r in relation.relations:
        x += evaluate_relation(r, weight_bias_dict)
    return weight_bias_dict[Relation.OpCode.keys()[relation.op]]["weight"] * x + weight_bias_dict[Relation.OpCode.keys()[relation.op]]["bias"]


def random_relation(max_depth):
    def build(relation, max_depth):
        if max_depth == 0 or random.uniform(0, 1) < 1.0 / 3.0:
            relation.rows = random.choice(range(1000, 2000))
            relation.width = random.choice(range(50, 100))
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
            # build(relation.relations, max_depth - 1)

    while True:
        relation = Relation()
        build(relation, max_depth)
        if relation.op != Relation.NONE:
            break

    return relation


def make_random_relation():
    max_depth = random.choice(range(1, 10))
    relation = random_relation(max_depth)

    e = []
    e.append(evaluate_relation(relation, system1_weight_bias_dict))
    e.append(evaluate_relation(relation, system2_weight_bias_dict))
    e.append(evaluate_relation(relation, system3_weight_bias_dict))
    min_index = 0
    min_value = e[0]
    for i in range(len(e)):
        if e[i] < min_value:
            min_index = i
            min_value = e[i]

    relation.result = min_index
    return relation
    # json = json_format.MessageToJson(relation)
    # return json
    # return relation.SerializeToString()


def main(unused_argv):
    record_output = tf.python_io.TFRecordWriter(FLAGS.output_path)
    for _ in xrange(FLAGS.num_samples):
        relation = make_random_relation()
        if FLAGS.output_format == "json":
            record_output.write(json_format.MessageToJson(relation))
        else:
            record_output.write(relation.SerializeToString())

    record_output.close()


if __name__ == '__main__':
    tf.app.run()
