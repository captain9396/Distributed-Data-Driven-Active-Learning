from pyspark import  SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint





conf = SparkConf().setAppName("labeledPoints")
sc = SparkContext(conf=conf)

row = ['2596,51,3,258,0,510']
rowRDD = sc.parallelize(row)



labeledPointsRDD = rowRDD.map(lambda _ : _.split(',')).map(lambda _ : LabeledPoint(_[-1], _[:-2]))
labeledPointsRDD.collect()