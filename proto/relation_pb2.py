# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: relation.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)




DESCRIPTOR = _descriptor.FileDescriptor(
  name='relation.proto',
  package='qpe.proto',
  serialized_pb='\n\x0erelation.proto\x12\tqpe.proto\"\xc7\x16\n\x08Relation\x12&\n\x02op\x18\x01 \x01(\x0e\x32\x1a.qpe.proto.Relation.OpCode\x12&\n\trelations\x18\x02 \x03(\x0b\x32\x13.qpe.proto.Relation\x12\x10\n\x08rowCount\x18\x03 \x01(\x01\x12:\n\x0e\x63umulativeCost\x18\x04 \x01(\x0b\x32\".qpe.proto.Relation.CumulativeCost\x12\x0e\n\x06result\x18\x05 \x01(\x03\x1a\x37\n\x0e\x43umulativeCost\x12\x0c\n\x04rows\x18\x01 \x01(\x01\x12\x0b\n\x03\x63pu\x18\x02 \x01(\x01\x12\n\n\x02io\x18\x03 \x01(\x01\"\xd3\x14\n\x06OpCode\x12\x15\n\x11\x41\x62stractConverter\x10\x00\x12\x13\n\x0f\x41\x62stractRelNode\x10\x01\x12\r\n\tAggregate\x10\x02\x12\x15\n\x11\x41ltTraitConverter\x10\x03\x12\x15\n\x11\x42indableAggregate\x10\x04\x12\x12\n\x0e\x42indableFilter\x10\x05\x12\x10\n\x0c\x42indableJoin\x10\x06\x12\x13\n\x0f\x42indableProject\x10\x07\x12\x0f\n\x0b\x42indableRel\x10\x08\x12\x10\n\x0c\x42indableSort\x10\t\x12\x15\n\x11\x42indableTableScan\x10\n\x12\x11\n\rBindableUnion\x10\x0b\x12\x12\n\x0e\x42indableValues\x10\x0c\x12\x12\n\x0e\x42indableWindow\x10\r\x12\t\n\x05\x42iRel\x10\x0e\x12\r\n\tBridgeRel\x10\x0f\x12\x08\n\x04\x43\x61lc\x10\x10\x12\x07\n\x03\x43hi\x10\x11\x12\x0b\n\x07\x43ollect\x10\x12\x12\r\n\tConverter\x10\x13\x12\x11\n\rConverterImpl\x10\x14\x12\r\n\tCorrelate\x10\x15\x12\t\n\x05\x44\x65lta\x10\x16\x12\x17\n\x13\x45lasticsearchFilter\x10\x17\x12\x18\n\x14\x45lasticsearchProject\x10\x18\x12\x14\n\x10\x45lasticsearchRel\x10\x19\x12\x15\n\x11\x45lasticsearchSort\x10\x1a\x12\x1a\n\x16\x45lasticsearchTableScan\x10\x1b\x12&\n\"ElasticsearchToEnumerableConverter\x10\x1c\x12\x17\n\x13\x45numerableAggregate\x10\x1d\x12\x16\n\x12\x45numerableBindable\x10\x1e\x12\x12\n\x0e\x45numerableCalc\x10\x1f\x12\x15\n\x11\x45numerableCollect\x10 \x12\x17\n\x13\x45numerableCorrelate\x10!\x12\x14\n\x10\x45numerableFilter\x10\"\x12\x1b\n\x17\x45numerableInterpretable\x10#\x12\x19\n\x15\x45numerableInterpreter\x10$\x12\x17\n\x13\x45numerableIntersect\x10%\x12\x12\n\x0e\x45numerableJoin\x10&\x12\x13\n\x0f\x45numerableLimit\x10\'\x12\x17\n\x13\x45numerableMergeJoin\x10(\x12\x13\n\x0f\x45numerableMinus\x10)\x12\x15\n\x11\x45numerableProject\x10*\x12\x11\n\rEnumerableRel\x10+\x12\x16\n\x12\x45numerableSemiJoin\x10,\x12\x12\n\x0e\x45numerableSort\x10-\x12\x1f\n\x1b\x45numerableTableFunctionScan\x10.\x12\x19\n\x15\x45numerableTableModify\x10/\x12\x17\n\x13\x45numerableTableScan\x10\x30\x12\x17\n\x13\x45numerableThetaJoin\x10\x31\x12\x17\n\x13\x45numerableUncollect\x10\x32\x12\x13\n\x0f\x45numerableUnion\x10\x33\x12\x14\n\x10\x45numerableValues\x10\x34\x12\x14\n\x10\x45numerableWindow\x10\x35\x12\x0c\n\x08\x45quiJoin\x10\x36\x12\x0c\n\x08\x45xchange\x10\x37\x12\n\n\x06\x46ilter\x10\x38\x12\n\n\x06\x46ooRel\x10\x39\x12\x10\n\x0cHepRelVertex\x10:\x12\x14\n\x10IntermediateNode\x10;\x12\x1a\n\x16InterpretableConverter\x10<\x12\x14\n\x10InterpretableRel\x10=\x12\r\n\tIntersect\x10>\x12\x11\n\rIterMergedRel\x10?\x12\x11\n\rIterSingleRel\x10@\x12\x11\n\rJdbcAggregate\x10\x41\x12\x0c\n\x08JdbcCalc\x10\x42\x12\x0e\n\nJdbcFilter\x10\x43\x12\x11\n\rJdbcIntersect\x10\x44\x12\x0c\n\x08JdbcJoin\x10\x45\x12\r\n\tJdbcMinus\x10\x46\x12\x0f\n\x0bJdbcProject\x10G\x12\x0b\n\x07JdbcRel\x10H\x12\x0c\n\x08JdbcSort\x10I\x12\x13\n\x0fJdbcTableModify\x10J\x12\x11\n\rJdbcTableScan\x10K\x12\x1d\n\x19JdbcToEnumerableConverter\x10L\x12\r\n\tJdbcUnion\x10M\x12\x0e\n\nJdbcValues\x10N\x12\x08\n\x04Join\x10O\x12\x0b\n\x07LeafRel\x10P\x12\x14\n\x10LogicalAggregate\x10Q\x12\x0f\n\x0bLogicalCalc\x10R\x12\x0e\n\nLogicalChi\x10S\x12\x14\n\x10LogicalCorrelate\x10T\x12\x10\n\x0cLogicalDelta\x10U\x12\x13\n\x0fLogicalExchange\x10V\x12\x11\n\rLogicalFilter\x10W\x12\x14\n\x10LogicalIntersect\x10X\x12\x0f\n\x0bLogicalJoin\x10Y\x12\x10\n\x0cLogicalMatch\x10Z\x12\x10\n\x0cLogicalMinus\x10[\x12\x12\n\x0eLogicalProject\x10\\\x12\x0f\n\x0bLogicalSort\x10]\x12\x1c\n\x18LogicalTableFunctionScan\x10^\x12\x16\n\x12LogicalTableModify\x10_\x12\x14\n\x10LogicalTableScan\x10`\x12\x10\n\x0cLogicalUnion\x10\x61\x12\x11\n\rLogicalValues\x10\x62\x12\x11\n\rLogicalWindow\x10\x63\x12\t\n\x05Match\x10\x64\x12\t\n\x05Minus\x10\x65\x12\x15\n\x11MockJdbcTableScan\x10\x66\x12\r\n\tMultiJoin\x10g\x12\t\n\x05MyRel\x10h\x12\x11\n\rNoneConverter\x10i\x12\x0f\n\x0bNoneLeafRel\x10j\x12\x11\n\rNoneSingleRel\x10k\x12\x08\n\x04Phys\x10l\x12\x0b\n\x07PhysAgg\x10m\x12\x10\n\x0cPhysicalSort\x10n\x12\x0f\n\x0bPhysLeafRel\x10o\x12\x0c\n\x08PhysProj\x10p\x12\x11\n\rPhysSingleRel\x10q\x12\x0c\n\x08PhysSort\x10r\x12\r\n\tPhysTable\x10s\x12\x1b\n\x17PhysToIteratorConverter\x10t\x12\x0b\n\x07Project\x10u\x12\x13\n\x0fRandomSingleRel\x10v\x12\r\n\tRelSubset\x10w\x12\x11\n\rRootSingleRel\x10x\x12\n\n\x06Sample\x10y\x12\x15\n\x11SelfFlatteningRel\x10z\x12\x0c\n\x08SemiJoin\x10{\x12\t\n\x05SetOp\x10|\x12\r\n\tSingleRel\x10}\x12\x14\n\x10SingletonLeafRel\x10~\x12\x08\n\x04Sort\x10\x7f\x12\x11\n\x0cSortExchange\x10\x80\x01\x12\x12\n\rStarTableScan\x10\x81\x01\x12\x16\n\x11TableFunctionScan\x10\x82\x01\x12\x10\n\x0bTableModify\x10\x83\x01\x12\x0e\n\tTableScan\x10\x84\x01\x12\x10\n\x0bTestLeafRel\x10\x85\x01\x12\x12\n\rTestSingleRel\x10\x86\x01\x12\x0e\n\tUncollect\x10\x87\x01\x12\n\n\x05Union\x10\x88\x01\x12\x0b\n\x06Values\x10\x89\x01\x12\x0b\n\x06Window\x10\x8a\x01\x42 \n\x0e\x65\x64u.ucr.cs.qpeB\x0eRelationProtos')



_RELATION_OPCODE = _descriptor.EnumDescriptor(
  name='OpCode',
  full_name='qpe.proto.Relation.OpCode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AbstractConverter', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AbstractRelNode', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Aggregate', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AltTraitConverter', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BindableAggregate', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BindableFilter', index=5, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BindableJoin', index=6, number=6,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BindableProject', index=7, number=7,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BindableRel', index=8, number=8,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BindableSort', index=9, number=9,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BindableTableScan', index=10, number=10,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BindableUnion', index=11, number=11,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BindableValues', index=12, number=12,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BindableWindow', index=13, number=13,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BiRel', index=14, number=14,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BridgeRel', index=15, number=15,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Calc', index=16, number=16,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Chi', index=17, number=17,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Collect', index=18, number=18,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Converter', index=19, number=19,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ConverterImpl', index=20, number=20,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Correlate', index=21, number=21,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Delta', index=22, number=22,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ElasticsearchFilter', index=23, number=23,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ElasticsearchProject', index=24, number=24,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ElasticsearchRel', index=25, number=25,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ElasticsearchSort', index=26, number=26,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ElasticsearchTableScan', index=27, number=27,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ElasticsearchToEnumerableConverter', index=28, number=28,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableAggregate', index=29, number=29,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableBindable', index=30, number=30,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableCalc', index=31, number=31,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableCollect', index=32, number=32,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableCorrelate', index=33, number=33,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableFilter', index=34, number=34,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableInterpretable', index=35, number=35,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableInterpreter', index=36, number=36,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableIntersect', index=37, number=37,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableJoin', index=38, number=38,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableLimit', index=39, number=39,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableMergeJoin', index=40, number=40,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableMinus', index=41, number=41,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableProject', index=42, number=42,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableRel', index=43, number=43,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableSemiJoin', index=44, number=44,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableSort', index=45, number=45,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableTableFunctionScan', index=46, number=46,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableTableModify', index=47, number=47,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableTableScan', index=48, number=48,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableThetaJoin', index=49, number=49,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableUncollect', index=50, number=50,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableUnion', index=51, number=51,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableValues', index=52, number=52,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EnumerableWindow', index=53, number=53,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EquiJoin', index=54, number=54,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Exchange', index=55, number=55,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Filter', index=56, number=56,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FooRel', index=57, number=57,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HepRelVertex', index=58, number=58,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='IntermediateNode', index=59, number=59,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='InterpretableConverter', index=60, number=60,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='InterpretableRel', index=61, number=61,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Intersect', index=62, number=62,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='IterMergedRel', index=63, number=63,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='IterSingleRel', index=64, number=64,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JdbcAggregate', index=65, number=65,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JdbcCalc', index=66, number=66,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JdbcFilter', index=67, number=67,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JdbcIntersect', index=68, number=68,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JdbcJoin', index=69, number=69,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JdbcMinus', index=70, number=70,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JdbcProject', index=71, number=71,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JdbcRel', index=72, number=72,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JdbcSort', index=73, number=73,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JdbcTableModify', index=74, number=74,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JdbcTableScan', index=75, number=75,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JdbcToEnumerableConverter', index=76, number=76,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JdbcUnion', index=77, number=77,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JdbcValues', index=78, number=78,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Join', index=79, number=79,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LeafRel', index=80, number=80,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalAggregate', index=81, number=81,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalCalc', index=82, number=82,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalChi', index=83, number=83,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalCorrelate', index=84, number=84,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalDelta', index=85, number=85,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalExchange', index=86, number=86,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalFilter', index=87, number=87,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalIntersect', index=88, number=88,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalJoin', index=89, number=89,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalMatch', index=90, number=90,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalMinus', index=91, number=91,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalProject', index=92, number=92,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalSort', index=93, number=93,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalTableFunctionScan', index=94, number=94,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalTableModify', index=95, number=95,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalTableScan', index=96, number=96,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalUnion', index=97, number=97,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalValues', index=98, number=98,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LogicalWindow', index=99, number=99,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Match', index=100, number=100,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Minus', index=101, number=101,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MockJdbcTableScan', index=102, number=102,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MultiJoin', index=103, number=103,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MyRel', index=104, number=104,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NoneConverter', index=105, number=105,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NoneLeafRel', index=106, number=106,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NoneSingleRel', index=107, number=107,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Phys', index=108, number=108,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PhysAgg', index=109, number=109,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PhysicalSort', index=110, number=110,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PhysLeafRel', index=111, number=111,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PhysProj', index=112, number=112,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PhysSingleRel', index=113, number=113,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PhysSort', index=114, number=114,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PhysTable', index=115, number=115,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PhysToIteratorConverter', index=116, number=116,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Project', index=117, number=117,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RandomSingleRel', index=118, number=118,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RelSubset', index=119, number=119,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RootSingleRel', index=120, number=120,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Sample', index=121, number=121,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SelfFlatteningRel', index=122, number=122,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SemiJoin', index=123, number=123,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SetOp', index=124, number=124,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SingleRel', index=125, number=125,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SingletonLeafRel', index=126, number=126,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Sort', index=127, number=127,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SortExchange', index=128, number=128,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='StarTableScan', index=129, number=129,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TableFunctionScan', index=130, number=130,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TableModify', index=131, number=131,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TableScan', index=132, number=132,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TestLeafRel', index=133, number=133,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TestSingleRel', index=134, number=134,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Uncollect', index=135, number=135,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Union', index=136, number=136,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Values', index=137, number=137,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Window', index=138, number=138,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=274,
  serialized_end=2917,
)


_RELATION_CUMULATIVECOST = _descriptor.Descriptor(
  name='CumulativeCost',
  full_name='qpe.proto.Relation.CumulativeCost',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='rows', full_name='qpe.proto.Relation.CumulativeCost.rows', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cpu', full_name='qpe.proto.Relation.CumulativeCost.cpu', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='io', full_name='qpe.proto.Relation.CumulativeCost.io', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=216,
  serialized_end=271,
)

_RELATION = _descriptor.Descriptor(
  name='Relation',
  full_name='qpe.proto.Relation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='op', full_name='qpe.proto.Relation.op', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='relations', full_name='qpe.proto.Relation.relations', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rowCount', full_name='qpe.proto.Relation.rowCount', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cumulativeCost', full_name='qpe.proto.Relation.cumulativeCost', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='result', full_name='qpe.proto.Relation.result', index=4,
      number=5, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_RELATION_CUMULATIVECOST, ],
  enum_types=[
    _RELATION_OPCODE,
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=30,
  serialized_end=2917,
)

_RELATION_CUMULATIVECOST.containing_type = _RELATION;
_RELATION.fields_by_name['op'].enum_type = _RELATION_OPCODE
_RELATION.fields_by_name['relations'].message_type = _RELATION
_RELATION.fields_by_name['cumulativeCost'].message_type = _RELATION_CUMULATIVECOST
_RELATION_OPCODE.containing_type = _RELATION;
DESCRIPTOR.message_types_by_name['Relation'] = _RELATION

class Relation(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType

  class CumulativeCost(_message.Message):
    __metaclass__ = _reflection.GeneratedProtocolMessageType
    DESCRIPTOR = _RELATION_CUMULATIVECOST

    # @@protoc_insertion_point(class_scope:qpe.proto.Relation.CumulativeCost)
  DESCRIPTOR = _RELATION

  # @@protoc_insertion_point(class_scope:qpe.proto.Relation)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), '\n\016edu.ucr.cs.qpeB\016RelationProtos')
# @@protoc_insertion_point(module_scope)
