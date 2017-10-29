from proto.relation_pb2 import Relation
import make_dataset


def main():
    relation = Relation()
    relation.op = Relation.NONE
    relation.rows = 10
    relation2 = Relation()
    relation2.op = Relation.EXTRACT
    relation2.relations.extend([relation, relation])
    relation3 = Relation()
    relation3.op = Relation.JOIN
    relation3.relations.extend([relation2, relation2, relation])

    print(relation2)
    print(Relation.OpCode.keys()[relation2.op])
    e = make_dataset.evaluate_relation(relation2, make_dataset.system1_weight_bias_dict)
    print(e)

    print(relation3)
    print(Relation.OpCode.keys()[relation3.op])
    e = make_dataset.evaluate_relation(relation3, make_dataset.system1_weight_bias_dict)
    print(e)


if __name__ == '__main__':
    main()
