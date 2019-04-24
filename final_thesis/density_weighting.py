from pyspark.sql import SparkSession
from debugger import Debugger
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







''' tunable parameters '''
n_samples = 5000
window_size = 10
n_estimators = 10
beta = 1










''' for striatum data only '''
train = sc.textFile('hdfs://node1:9000/input/striatum_train_mini.txt')

train = sc.parallelize(train.take(n_samples))

train = train.map(lambda _ : _.strip().split())
train = train.map(lambda _ : LabeledPoint(0 if int(_[-1]) == -1 else 1, np.array(_[:-1]).astype(float)))
test = sc.textFile('hdfs://node1:9000/input/striatum_test_mini.txt')
test = test.map(lambda _ : _.strip().split())
test = test.map(lambda _ : LabeledPoint(0 if int(_[-1]) == -1 else 1, np.array(_[:-1]).astype(float)))




''' proximity matrix construction '''
data = sc.textFile('hdfs://node1:9000/input/striatum_train_mini.txt')

''' 1000 ta nilam time kom '''
data = sc.parallelize(data.take(n_samples))


data = data.map(lambda _ : np.array(_.strip().split()[:-1]).astype(float))
data = data.map(lambda _ : _/np.linalg.norm(_))
U = data.zipWithIndex().map(lambda _ : IndexedRow(_[1], _[0]))
U = IndexedRowMatrix(U)
UT = U.toCoordinateMatrix()
UT = UT.transpose()
U = U.toBlockMatrix()
UT = UT.toBlockMatrix()
S = U.multiply(UT)
S_coord = S.toCoordinateMatrix()
similarities = S_coord.entries


print('matrix done!')




train = train.zipWithIndex()
keyfirst_train = train.map(lambda _: (_[1], _[0]))
n_total = train.count()
print('n_total = ', n_total)
print('sim = ', similarities.count())

labeled_indices = sc.parallelize([(x, 0) for x in range(0, window_size)])
unlabeled_indices = sc.parallelize([(x, 0) for x in range(window_size, n_total)])

similarities = similarities.map(lambda _: (_.i, _.j, _.value))


''' for the sake of query planning we are taking an empty rdd '''
rdd = sc.parallelize([])
for li in labeled_indices.collect():
    mark_labeled = li[0]
    rdd = rdd.union(similarities.filter(lambda _: _[0] == mark_labeled or _[1] == mark_labeled))
similarities = similarities.subtract(rdd)





cnt = 1

''' loop producing labelling sequence '''
while True:
    labeled_data = labeled_indices.leftOuterJoin(keyfirst_train).map(lambda _: (_[1][1], _[0]))
    unlabeled_data = unlabeled_indices.leftOuterJoin(keyfirst_train).map(lambda _: (_[1][1], _[0]))

    print('labeled = ', labeled_indices.count(), ' unlabeled = ', unlabeled_indices.count())

    if unlabeled_indices.isEmpty() :
        break


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

    ''' real entropies are taken ; diffierent than US implementation '''
    entropies = classPrediction.map(lambda _: - (1-(_[1]/n_estimators)) * np.log2((1-(_[1]/n_estimators))))

    ''' base strategy => uncertainty sampling '''
    unlabeled_entropies = unlabeled_indices.map(lambda _: _[0])\
        .zipWithIndex()\
        .map(lambda _: (_[1], _[0]))\
        .leftOuterJoin(entropies.zipWithIndex().map(lambda _:(_[1], _[0])))\
        .map(lambda _:_[1])

    ''' similarity calculation using proximity matrix values '''
    unlabeled_similarities = unlabeled_indices.leftOuterJoin(similarities.map(lambda _:(_[0] , _[2])))\
        .map(lambda _:(_[0], _[1][0]+_[1][1]))\
        .groupByKey()\
        .mapValues(sum)




    ''' information density values  '''
    unlabeled_heuristic_values = unlabeled_entropies.leftOuterJoin(unlabeled_similarities).map(lambda _:(_[0], _[1][0]*_[1][1]))
    sorted_heuristic_values = unlabeled_heuristic_values.sortBy(lambda _: _[1],ascending=False)

    print(sorted_heuristic_values.take(10))
    ''' taking chunk of samples with maximum heuristic value '''
    add_to_labeled_set = sc.parallelize(sorted_heuristic_values.take(window_size))

    ''' updating the unlabeled and labeled indices sets '''
    unlabeled_indices = unlabeled_indices.subtractByKey(add_to_labeled_set)
    labeled_indices = labeled_indices.union(add_to_labeled_set.map(lambda _: (_[0], 0)))

    print('Iteration ', cnt, ' -- accu = ', (1- (testErr.count() / test.count()))*100)
    cnt += 1


debug.TIMESTAMP(2)
