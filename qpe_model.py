from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_fold.public.blocks as td


NUM_LABELS = 3


def preprocess_relation(expr):
    # Set the op field for leaf nodes, so we can handle cases uniformly.
    if expr['rows'] is not None:
        expr['op'] = {'name': 'NONE'}
    return expr


class QueryPerformanceEstimatorModel(object):

    def __init__(self, state_size):
        relation_declaration = td.ForwardDeclaration(td.PyObjectType(), state_size)

        nrows = (td.GetItem('rows') >> td.Scalar(dtype='int32') >>
                 td.Function(td.Embedding(10, state_size, name='terminal_embed')))

        def query_op(name):
            return (td.GetItem('relations')
                    >> td.Map(relation_declaration())
                    >> td.Fold(td.Concat() >> td.Function(td.FC(state_size, name='FC_' + name)), td.FromTensor(tf.zeros(state_size))))
            # return (td.Record({'relations', td.Map(relation_declaration())})
            #         >> td.Concat()
            #         >> td.FC(state_size, name='FC_' + name))

        cases = td.OneOf(lambda x: x['op']['name'], {
            'NONE': nrows,
            'JOIN': query_op('JOIN'),
            'MAPJOIN': query_op('MAPJOIN'),
            'EXTRACT': query_op('EXTRACT'),
            'FILTER': query_op('FILTER'),
            'FORWARD': query_op('FORWARD'),
            'GROUPBY': query_op('GROUPBY'),
            'LIMIT': query_op('LIMIT'),
            'SCRIPT': query_op('SCRIPT'),
            'SELECT': query_op('SELECT'),
            'TABLESCAN': query_op('TABLESCAN'),
            'FILESINK' : query_op('FILESINK'),
            'REDUCESINK': query_op('REDUCESINK'),
            'UNION': query_op('UNION'),
            'UDTF': query_op('UDTF'),
            'LATERALVIEWJOIN': query_op('LATERALVIEWJOIN'),
            'LATERALVIEWFORWARD' : query_op('LATERALVIEWFORWARD'),
            'HASHTABLESINK': query_op('HASHTABLESINK'),
            'HASHTABLEDUMMY': query_op('HASHTABLEDUMMY'),
            'PTF': query_op('PTF'),
            'MUX': query_op('MUX'),
            'DEMUX': query_op('DEMUX'),
            'EVENT': query_op('EVENT'),
            'ORCFILEMERGE': query_op('ORCFILEMERGE'),
            'RCFILEMERGE': query_op('RCFILEMERGE'),
            'MERGEJOIN': query_op('MERGEJOIN'),
            'SPARKPRUNINGSINK': query_op('SPARKPRUNINGSINK')})

        relation = td.InputTransform(preprocess_relation) >> cases
        relation_declaration.resolve_to(relation)
        relation_logits = (relation >> td.FC(NUM_LABELS, activation=None, name='FC_logits'))
        relation_label = (td.GetItem('result') >> td.OneHot(NUM_LABELS))
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
