from pyspark.sql import SparkSession
from debugger import Debugger
from vector_accum import VectorAccumulatorParam
from pyspark import  SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.tree import DecisionTreeModel
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
data = sc.parallelize(data.take(15000))
train, test = data.randomSplit([50.0, 50.0])




window_size = 500

train = train.zipWithIndex()
keyfirst_train = train.map(lambda _: (_[1], _[0]))
n_total = train.count()


labeled_indices = sc.parallelize([(x, None) for x in range(0, window_size)])
unlabeled_indices = sc.parallelize([(x, None) for x in range(window_size, n_total)])




cnt = 1
while True:

    labeled_data = labeled_indices.leftOuterJoin(keyfirst_train).map(lambda _: (_[1][1], _[0]))
    unlabeled_data = unlabeled_indices.leftOuterJoin(keyfirst_train).map(lambda _: (_[1][1], _[0]))

    print('labeled = ', labeled_indices.count(), ' unlabeled = ', unlabeled_indices.count())

    if unlabeled_indices.isEmpty() :
        break

    n_estimators = 10
    model = RandomForest.trainClassifier(labeled_data.map(lambda _: _[0]),
                                         numClasses=2,
                                         categoricalFeaturesInfo={},
                                         numTrees=n_estimators,
                                         featureSubsetStrategy="auto",
                                         impurity='gini')


    ''' accuracy test on testset here'''
    predictions = model.predict(test.map(lambda x: x.features))
    labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(lambda _ : _[0] != _[1])


    n_unlabeled = unlabeled_data.count()


    rdd = sc.parallelize([])
    for tree in model._java_model.trees():
        predX = DecisionTreeModel(tree).predict(unlabeled_data.map(lambda _ : _[0].features))\
            .zipWithIndex()\
            .map(lambda _: (_[1], _[0]))
        rdd = rdd.union(predX)


    classPrediction = rdd.groupByKey().mapValues(sum)
    classPrediction = classPrediction.sortByKey()
    entropies = classPrediction.map(lambda _: abs(0.5 - (1-(_[1]/n_estimators))))

    unlabeled_entropies = unlabeled_indices.map(lambda _: _[0])\
        .zipWithIndex()\
        .map(lambda _: (_[1], _[0]))\
        .leftOuterJoin(entropies.zipWithIndex().map(lambda _:(_[1], _[0])))\
        .map(lambda _:_[1])

    sorted_unlabeled_entropies = unlabeled_entropies.sortBy(lambda _: _[1])


    add_to_labeled_set = sc.parallelize(sorted_unlabeled_entropies.take(window_size))

    unlabeled_indices = unlabeled_indices.subtractByKey(add_to_labeled_set)
    labeled_indices = labeled_indices.union(add_to_labeled_set.map(lambda _: (_[0], None)))
    print('Iteration ', cnt, ' -- accu = ', (1- (testErr.count() / test.count()))*100)
    cnt += 1


debug.TIMESTAMP(2)








