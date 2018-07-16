from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf
import random
import os
import csv
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
        return relation.rowCount
    x = 0
    for r in relation.relations:
        x += evaluate_relation(r, weight_bias_dict)
    return weight_bias_dict[Relation.OpCode.keys()[relation.op]]["weight"] * x + weight_bias_dict[Relation.OpCode.keys()[relation.op]]["bias"]


def random_relation(max_depth):
    def build(relation, max_depth):
        if max_depth == 0 or random.uniform(0, 1) < 1.0 / 3.0:
            relation.rowCount = random.choice(range(1000, 2000))
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

    # e = []
    # e.append(evaluate_relation(relation, system1_weight_bias_dict))
    # e.append(evaluate_relation(relation, system2_weight_bias_dict))
    # e.append(evaluate_relation(relation, system3_weight_bias_dict))
    # min_index = 0
    # min_value = e[0]
    # for i in range(len(e)):
    #     if e[i] < min_value:
    #         min_index = i
    #         min_value = e[i]

    relation.result = 1
    return relation
    # json = json_format.MessageToJson(relation)
    # return json
    # return relation.SerializeToString()


def convert_relation_to_vector(relation, json_content):
    dict = {'AbstractConverter': 0,
            'AbstractRelNode': 0,
            'Aggregate': 0,
            'AltTraitConverter': 0,
            'BindableAggregate': 0,
            'BindableFilter': 0,
            'BindableJoin': 0,
            'BindableProject': 0,
            'BindableRel': 0,
            'BindableSort': 0,
            'BindableTableScan': 0,
            'BindableUnion': 0,
            'BindableValues': 0,
            'BindableWindow': 0,
            'BiRel': 0,
            'BridgeRel': 0,
            'Calc': 0,
            'Chi': 0,
            'Collect': 0,
            'Converter': 0,
            'ConverterImpl': 0,
            'Correlate': 0,
            'Delta': 0,
            'ElasticsearchFilter': 0,
            'ElasticsearchProject': 0,
            'ElasticsearchRel': 0,
            'ElasticsearchSort': 0,
            'ElasticsearchTableScan': 0,
            'ElasticsearchToEnumerableConverter': 0,
            'EnumerableAggregate': 0,
            'EnumerableBindable': 0,
            'EnumerableCalc': 0,
            'EnumerableCollect': 0,
            'EnumerableCorrelate': 0,
            'EnumerableFilter': 0,
            'EnumerableInterpretable': 0,
            'EnumerableInterpreter': 0,
            'EnumerableIntersect': 0,
            'EnumerableJoin': 0,
            'EnumerableLimit': 0,
            'EnumerableMergeJoin': 0,
            'EnumerableMinus': 0,
            'EnumerableProject': 0,
            'EnumerableRel': 0,
            'EnumerableSemiJoin': 0,
            'EnumerableSort': 0,
            'EnumerableTableFunctionScan': 0,
            'EnumerableTableModify': 0,
            'EnumerableTableScan': 0,
            'EnumerableThetaJoin': 0,
            'EnumerableUncollect': 0,
            'EnumerableUnion': 0,
            'EnumerableValues': 0,
            'EnumerableWindow': 0,
            'EquiJoin': 0,
            'Exchange': 0,
            'Filter': 0,
            'FooRel': 0,
            'HepRelVertex': 0,
            'IntermediateNode': 0,
            'InterpretableConverter': 0,
            'InterpretableRel': 0,
            'Intersect': 0,
            'IterMergedRel': 0,
            'IterSingleRel': 0,
            'JdbcAggregate': 0,
            'JdbcCalc': 0,
            'JdbcFilter': 0,
            'JdbcIntersect': 0,
            'JdbcJoin': 0,
            'JdbcMinus': 0,
            'JdbcProject': 0,
            'JdbcRel': 0,
            'JdbcSort': 0,
            'JdbcTableModify': 0,
            'JdbcTableScan': 0,
            'JdbcToEnumerableConverter': 0,
            'JdbcUnion': 0,
            'JdbcValues': 0,
            'Join': 0,
            'LeafRel': 0,
            'LogicalAggregate': 0,
            'LogicalCalc': 0,
            'LogicalChi': 0,
            'LogicalCorrelate': 0,
            'LogicalDelta': 0,
            'LogicalExchange': 0,
            'LogicalFilter': 0,
            'LogicalIntersect': 0,
            'LogicalJoin': 0,
            'LogicalMatch': 0,
            'LogicalMinus': 0,
            'LogicalProject': 0,
            'LogicalSort': 0,
            'LogicalTableFunctionScan': 0,
            'LogicalTableModify': 0,
            'LogicalTableScan': 0,
            'LogicalUnion': 0,
            'LogicalValues': 0,
            'LogicalWindow': 0,
            'Match': 0,
            'Minus': 0,
            'MockJdbcTableScan': 0,
            'MultiJoin': 0,
            'MyRel': 0,
            'NoneConverter': 0,
            'NoneLeafRel': 0,
            'NoneSingleRel': 0,
            'Phys': 0,
            'PhysAgg': 0,
            'PhysicalSort': 0,
            'PhysLeafRel': 0,
            'PhysProj': 0,
            'PhysSingleRel': 0,
            'PhysSort': 0,
            'PhysTable': 0,
            'PhysToIteratorConverter': 0,
            'Project': 0,
            'RandomSingleRel': 0,
            'RelSubset': 0,
            'RootSingleRel': 0,
            'Sample': 0,
            'SelfFlatteningRel': 0,
            'SemiJoin': 0,
            'SetOp': 0,
            'SingleRel': 0,
            'SingletonLeafRel': 0,
            'Sort': 0,
            'SortExchange': 0,
            'StarTableScan': 0,
            'TableFunctionScan': 0,
            'TableModify': 0,
            'TableScan': 0,
            'TestLeafRel': 0,
            'TestSingleRel': 0,
            'Uncollect': 0,
            'Union': 0,
            'Values': 0,
            'Window': 0}
    for key in dict.keys():
        dict[key] = json_content.count(key)
        # dict[key] = get_value(relation, key)
    values = ""
    for key in sorted(dict):
        values += str(dict[key]) + ","
    values += str(relation.result)
    return dict, values


def get_value(relation, key):
    result = 0
    if key == Relation.OpCode.keys()[relation.op]:
        result += relation.rowCount
    if len(relation.relations) == 0:
        return result
    else:
        for r in relation.relations:
            result += get_value(r, key)
        return result


def main(unused_argv):
    train_record_output = tf.python_io.TFRecordWriter(FLAGS.output_path + "/train.dat")
    validation_record_output = tf.python_io.TFRecordWriter(FLAGS.output_path + "/validation.dat")
    test_record_output = tf.python_io.TFRecordWriter(FLAGS.output_path + "/test.dat")
    # train_record_output_csv = FLAGS.output_path + "/train.csv"
    # validation_record_output_csv = FLAGS.output_path + "/validation.csv"
    # test_record_output_csv = FLAGS.output_path + "/test.csv"
    # for _ in xrange(FLAGS.num_samples):
    #     relation = make_random_relation()
    #     if FLAGS.output_format == "json":
    #         train_record_output.write(json_format.MessageToJson(relation))
    #     else:
    #         train_record_output.write(relation.SerializeToString())

    train_dataset_path = './dataset/train'
    for filename in os.listdir(train_dataset_path):
        with open(train_dataset_path + "/" + filename) as json_query_file:
            json_content = json_query_file.read()
            relation = Relation()
            json_format.Parse(json_content, relation)
            # print(json_format.MessageToJson(relation))
            # train_record_output.write(json_format.MessageToJson(relation))
            train_record_output.write(relation.SerializeToString())
            # op_dict, values = convert_relation_to_vector(relation, json_content)
            # with open(train_record_output_csv, 'a') as csv_file:
            #     csv_file.write(values)
            #     csv_file.write("\n")

    train_record_output.close()

    validation_dataset_path = './dataset/validation'
    for filename in os.listdir(validation_dataset_path):
        with open(validation_dataset_path + "/" + filename) as json_query_file:
            json_content = json_query_file.read()
            relation = Relation()
            json_format.Parse(json_content, relation)
            # print(json_format.MessageToJson(relation))
            # train_record_output.write(json_format.MessageToJson(relation))
            validation_record_output.write(relation.SerializeToString())
            # op_dict, values = convert_relation_to_vector(relation, json_content)
            # with open(validation_record_output_csv, 'a') as csv_file:
            #     csv_file.write(values)
            #     csv_file.write("\n")

    validation_record_output.close()

    test_dataset_path = './dataset/test'
    for filename in os.listdir(test_dataset_path):
        with open(test_dataset_path + "/" + filename) as json_query_file:
            json_content = json_query_file.read()
            relation = Relation()
            json_format.Parse(json_content, relation)
            # print(json_format.MessageToJson(relation))
            # train_record_output.write(json_format.MessageToJson(relation))
            test_record_output.write(relation.SerializeToString())
            # op_dict, values = convert_relation_to_vector(relation, json_content)
            # with open(test_record_output_csv, 'a') as csv_file:
            #     csv_file.write(values)
            #     csv_file.write("\n")

    test_record_output.close()


if __name__ == '__main__':
    tf.app.run()
