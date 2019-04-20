from pyspark.sql import SparkSession
from debugger import Debugger
from pyspark import  SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from  pyspark.mllib.linalg import SparseVector, DenseVector
from sklearn.preprocessing import normalize
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix, BlockMatrix, MatrixEntry, RowMatrix, CoordinateMatrix
from pyspark.mllib.linalg import Matrix, Matrices, DenseMatrix
from pyspark.mllib.feature import Normalizer
import numpy as np
# setup spark context and config
conf = SparkConf().setAppName("labeledPoints")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

debug = Debugger()
debug.TIMESTAMP(1)
spark = SparkSession(sc)






data = sc.textFile('hdfs://node1:9000/input/apple_banana.txt')
data = data.map(lambda _ : _.strip().split())
data = data.map(lambda _ : LabeledPoint(int(_[-1]), np.array(_[:-1]).astype(float)))
train, test = data.randomSplit([80.0, 20.0])


train = train.zipWithIndex()
unlabeledIndices = train.map(lambda _ : _[1])
labeledIndices = sc.parallelize([])
n = train.count()


initdata = open('unlabeled_init.txt').readlines()
initdata = [line.strip().split() for line in initdata]
initdata = sc.parallelize([LabeledPoint(int(x[-1]), np.array(x[:-1]).astype(float)) for x in initdata])


model = RandomForest.trainClassifier(initdata, numClasses=2, categoricalFeaturesInfo={},numTrees=10, featureSubsetStrategy="auto",impurity='gini')
predictions = model.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda _ : _[0] != _[1])
print('Test Error : ' , testErr.count() / test.count())




#print(unlabeledIndices.take(200))


debug.TIMESTAMP(2)
