from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_fold.public.blocks as td


NUM_LABELS = 2


def preprocess_relation(expr):
    # Set the op field for leaf nodes, so we can handle cases uniformly.
    # if expr['rows'] is not None:
    #     expr['op'] = {'name': 'NONE'}
    # print(expr)
    return expr


def result_index(result):
    if result == 1:
        return 0
    if result == 2:
        return 1
    return 2


class QueryPerformanceEstimatorModel(object):

    def __init__(self, state_size):
        relation_declaration = td.ForwardDeclaration(td.PyObjectType(), state_size)

        nrows = (td.GetItem('rowCount') >> td.Scalar(dtype='int64') >> td.Function(tf.to_float) >> td.Function(tf.log) >> td.Function(tf.to_int32) >>
                 td.Function(td.Embedding(10, state_size, name='terminal_embed')))

        def query_op(name):
            return (td.GetItem('relations')
                    >> td.Map(relation_declaration())
                    >> td.Fold(td.Concat() >> td.Function(td.FC(state_size, name='FC_' + name)), td.FromTensor(tf.zeros(state_size))))
            # return (td.GetItem('relations')
            #         >> td.Map(relation_declaration())
            #         >> td.Fold(td.Concat() >> td.Function(td.RNN(op_cell, name='RNN_' + name)),
            #                    td.FromTensor(tf.zeros(state_size))))
            # return (td.Record({'relations', td.Map(relation_declaration())})
            #         >> td.Concat()
            #         >> td.FC(state_size, name='FC_' + name))

        cases = td.OneOf(lambda x: x['op']['name'], {
            'AbstractConverter': query_op('AbstractConverter'),
            'AbstractRelNode': query_op('AbstractRelNode'),
            'Aggregate': query_op('Aggregate'),
            'AltTraitConverter': query_op('AltTraitConverter'),
            'BindableAggregate': query_op('BindableAggregate'),
            'BindableFilter': query_op('BindableFilter'),
            'BindableJoin': query_op('BindableJoin'),
            'BindableProject': query_op('BindableProject'),
            'BindableRel': query_op('BindableRel'),
            'BindableSort': query_op('BindableSort'),
            'BindableTableScan': query_op('BindableTableScan'),
            'BindableUnion': query_op('BindableUnion'),
            'BindableValues': query_op('BindableValues'),
            'BindableWindow': query_op('BindableWindow'),
            'BiRel': query_op('BiRel'),
            'BridgeRel': query_op('BridgeRel'),
            'Calc': query_op('Calc'),
            'Chi': query_op('Chi'),
            'Collect': query_op('Collect'),
            'Converter': query_op('Converter'),
            'ConverterImpl': query_op('ConverterImpl'),
            'Correlate': query_op('Correlate'),
            'Delta': query_op('Delta'),
            'ElasticsearchFilter': query_op('ElasticsearchFilter'),
            'ElasticsearchProject': query_op('ElasticsearchProject'),
            'ElasticsearchRel': query_op('ElasticsearchRel'),
            'ElasticsearchSort': query_op('ElasticsearchSort'),
            'ElasticsearchTableScan': query_op('ElasticsearchTableScan'),
            'ElasticsearchToEnumerableConverter': query_op('ElasticsearchToEnumerableConverter'),
            'EnumerableAggregate': query_op('EnumerableAggregate'),
            'EnumerableBindable': query_op('EnumerableBindable'),
            'EnumerableCalc': query_op('EnumerableCalc'),
            'EnumerableCollect': query_op('EnumerableCollect'),
            'EnumerableCorrelate': query_op('EnumerableCorrelate'),
            'EnumerableFilter': query_op('EnumerableFilter'),
            'EnumerableInterpretable': query_op('EnumerableInterpretable'),
            'EnumerableInterpreter': query_op('EnumerableInterpreter'),
            'EnumerableIntersect': query_op('EnumerableIntersect'),
            'EnumerableJoin': query_op('EnumerableJoin'),
            'EnumerableLimit': query_op('EnumerableLimit'),
            'EnumerableMergeJoin': query_op('EnumerableMergeJoin'),
            'EnumerableMinus': query_op('EnumerableMinus'),
            'EnumerableProject': query_op('EnumerableProject'),
            'EnumerableRel': query_op('EnumerableRel'),
            'EnumerableSemiJoin': query_op('EnumerableSemiJoin'),
            'EnumerableSort': query_op('EnumerableSort'),
            'EnumerableTableFunctionScan': query_op('EnumerableTableFunctionScan'),
            'EnumerableTableModify': query_op('EnumerableTableModify'),
            'EnumerableTableScan': nrows,
            'EnumerableThetaJoin': query_op('EnumerableThetaJoin'),
            'EnumerableUncollect': query_op('EnumerableUncollect'),
            'EnumerableUnion': query_op('EnumerableUnion'),
            'EnumerableValues': query_op('EnumerableValues'),
            'EnumerableWindow': query_op('EnumerableWindow'),
            'EquiJoin': query_op('EquiJoin'),
            'Exchange': query_op('Exchange'),
            'Filter': query_op('Filter'),
            'FooRel': query_op('FooRel'),
            'HepRelVertex': query_op('HepRelVertex'),
            'IntermediateNode': query_op('IntermediateNode'),
            'InterpretableConverter': query_op('InterpretableConverter'),
            'InterpretableRel': query_op('InterpretableRel'),
            'Intersect': query_op('Intersect'),
            'IterMergedRel': query_op('IterMergedRel'),
            'IterSingleRel': query_op('IterSingleRel'),
            'JdbcAggregate': query_op('JdbcAggregate'),
            'JdbcCalc': query_op('JdbcCalc'),
            'JdbcFilter': query_op('JdbcFilter'),
            'JdbcIntersect': query_op('JdbcIntersect'),
            'JdbcJoin': query_op('JdbcJoin'),
            'JdbcMinus': query_op('JdbcMinus'),
            'JdbcProject': query_op('JdbcProject'),
            'JdbcRel': query_op('JdbcRel'),
            'JdbcSort': query_op('JdbcSort'),
            'JdbcTableModify': query_op('JdbcTableModify'),
            'JdbcTableScan': query_op('JdbcTableScan'),
            'JdbcToEnumerableConverter': query_op('JdbcToEnumerableConverter'),
            'JdbcUnion': query_op('JdbcUnion'),
            'JdbcValues': query_op('JdbcValues'),
            'Join': query_op('Join'),
            'LeafRel': query_op('LeafRel'),
            'LogicalAggregate': query_op('LogicalAggregate'),
            'LogicalCalc': query_op('LogicalCalc'),
            'LogicalChi': query_op('LogicalChi'),
            'LogicalCorrelate': query_op('LogicalCorrelate'),
            'LogicalDelta': query_op('LogicalDelta'),
            'LogicalExchange': query_op('LogicalExchange'),
            'LogicalFilter': query_op('LogicalFilter'),
            'LogicalIntersect': query_op('LogicalIntersect'),
            'LogicalJoin': query_op('LogicalJoin'),
            'LogicalMatch': query_op('LogicalMatch'),
            'LogicalMinus': query_op('LogicalMinus'),
            'LogicalProject': query_op('LogicalProject'),
            'LogicalSort': query_op('LogicalSort'),
            'LogicalTableFunctionScan': query_op('LogicalTableFunctionScan'),
            'LogicalTableModify': query_op('LogicalTableModify'),
            'LogicalTableScan': query_op('LogicalTableScan'),
            'LogicalUnion': query_op('LogicalUnion'),
            'LogicalValues': query_op('LogicalValues'),
            'LogicalWindow': query_op('LogicalWindow'),
            'Match': query_op('Match'),
            'Minus': query_op('Minus'),
            'MockJdbcTableScan': query_op(''),
            'MultiJoin': query_op('MockJdbcTableScan'),
            'MyRel': query_op('MyRel'),
            'NoneConverter': query_op('NoneConverter'),
            'NoneLeafRel': query_op('NoneLeafRel'),
            'NoneSingleRel': query_op('NoneSingleRel'),
            'Phys': query_op('Phys'),
            'PhysAgg': query_op('PhysAgg'),
            'PhysicalSort': query_op('PhysicalSort'),
            'PhysLeafRel': query_op('PhysLeafRel'),
            'PhysProj': query_op('PhysProj'),
            'PhysSingleRel': query_op('PhysSingleRel'),
            'PhysSort': query_op('PhysSort'),
            'PhysTable': query_op('PhysTable'),
            'PhysToIteratorConverter': query_op('PhysToIteratorConverter'),
            'Project': query_op('Project'),
            'RandomSingleRel': query_op('RandomSingleRel'),
            'RelSubset': query_op('RelSubset'),
            'RootSingleRel': query_op('RootSingleRel'),
            'Sample': query_op('Sample'),
            'SelfFlatteningRel': query_op('SelfFlatteningRel'),
            'SemiJoin': query_op('SemiJoin'),
            'SetOp': query_op('SetOp'),
            'SingleRel': query_op('SingleRel'),
            'SingletonLeafRel': query_op('SingletonLeafRel'),
            'Sort': query_op('Sort'),
            'SortExchange': query_op('SortExchange'),
            'StarTableScan': query_op('StarTableScan'),
            'TableFunctionScan': query_op('TableFunctionScan'),
            'TableModify': query_op('TableModify'),
            'TableScan': query_op('TableScan'),
            'TestLeafRel': query_op('TestLeafRel'),
            'TestSingleRel': query_op('TestSingleRel'),
            'Uncollect': query_op('Uncollect'),
            'Union': query_op('Union'),
            'Values': query_op('Values'),
            'Window': query_op('Window')})

        relation = td.InputTransform(preprocess_relation) >> cases
        relation_declaration.resolve_to(relation)
        relation_logits = (relation >> td.FC(state_size, activation=tf.sigmoid, name='FC_logits_1')
                           >> td.FC(state_size*4, activation=tf.sigmoid, name='FC_logits_2')
                           >> td.FC(state_size*4, activation=tf.sigmoid, name='FC_logits_3')
                           >> td.FC(state_size*4, activation=tf.sigmoid, name='FC_logits_4')
                           >> td.FC(state_size*4, activation=tf.sigmoid, name='FC_logits_5')
                           >> td.FC(state_size*4, activation=tf.sigmoid, name='FC_logits_6')
                           >> td.FC(state_size*4, activation=tf.sigmoid, name='FC_logits_7')
                           >> td.FC(state_size*4, activation=tf.sigmoid, name='FC_logits_8')
                           >> td.FC(state_size*4, activation=tf.sigmoid, name='FC_logits_9')
                           >> td.FC(NUM_LABELS, activation=tf.sigmoid, name='FC_logits_10'))
        relation_label = (td.GetItem('result') >> td.InputTransform(result_index) >> td.OneHot(NUM_LABELS))
        model = td.AllOf(relation_logits, relation_label)
        self._compiler = td.Compiler.create(model)
        (logits, labels) = self._compiler.output_tensors

        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        self._accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1)), dtype=tf.float32))

        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        optr = tf.train.GradientDescentOptimizer(0.01)
        self._train_op = optr.minimize(self._loss, global_step=self._global_step)


    @property
    def loss(self):
        return self._loss

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def train_op(self):
        return self._train_op

    @property
    def global_step(self):
        return self._global_step

    def build_feed_dict(self, relations):
        return self._compiler.build_feed_dict(relations)
