from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.feature import StandardScaler, StandardScalerModel
import datetime
from datetime import timedelta



# setup spark context and config
conf = SparkConf().setAppName("test")

# conf = SparkConf().setAppName("Print Elements of RDD")\
#     .setMaster("local[4]").set("spark.executor.memory","1g");

sc = SparkContext(conf=conf)


HDFS_DIRECTORY = "hdfs://node1:9000/input/"




'''
FILES in HDFS
--------------
'''
# fileDirectory = 'ITERATIVE_TREE_BIG.txt'
# fileDirectory = 'RANDOM_TREE_BIG.txt'
# fileDirectory = 'checkerboard2x2_test.txt'
# fileDirectory = 'checkerboard2x2_train.txt'
# fileDirectory = 'checkerboard4x4_test.txt'
# fileDirectory = 'checkerboard4x4_train.txt'
# fileDirectory = 'striatum_test_data.txt'
# fileDirectory = 'striatum_train_data.txt'




## here some changes needed to be made <<<<<<<<<<<



class Dataset:
    def __init__(self):
        # each dataset will have training and test data with label
        self.trainData = None
        self.trainLabels = None
        self.testData = None
        self.testLabels = None





class DatasetCheckerboard2x2(Dataset):

    def __init__(self):
        Dataset.__init__(self)

        trainDirectory =  HDFS_DIRECTORY + 'checkerboard2x2_train.txt'
        train = sc.textFile(trainDirectory)
        features = train.map(lambda _ : _.split(' ')[:-1])
        labels = train.map(lambda _: _.split(' ')[-1])
        scaler = StandardScaler(withMean=True, withStd=True).fit(features)
        self.trainSet = labels.zip(scaler.transform(features)).map(lambda _: LabeledPoint(_[0], _[1]))

        testDirectory = HDFS_DIRECTORY + 'checkerboard2x2_test.txt'
        test = sc.textFile(testDirectory)
        features = test.map(lambda _ : _.split(' ')[:-1])
        labels = test.map(lambda _: _.split(' ')[-1])
        scaler = StandardScaler(withMean=True, withStd=True).fit(features)
        self.testSet = labels.zip(scaler.transform(features)).map(lambda _: LabeledPoint(_[0], _[1]))




        # reg = RandomForest.trainRegressor(self.trainSet, numTrees=100, categoricalFeaturesInfo={})
        # predictions = reg.predict(self.testSet.map(lambda x: x.features))
        # labelsAndPredictions = self.testSet.map(lambda lp: lp.label).zip(predictions)
        # testMSE = labelsAndPredictions.map(lambda x: (x[0] - x[1]) ** 2).sum() / float(self.testSet.count())
        # print('Test Mean Squared Error = ' + str(testMSE))





class DatasetCheckerboard4x4(Dataset):

    def __init__(self):
        Dataset.__init__(self)

        trainDirectory =  HDFS_DIRECTORY + 'checkerboard4x4_train.txt'
        train = sc.textFile(trainDirectory)
        features = train.map(lambda _ : _.split(' ')[:-1])
        labels = train.map(lambda _: _.split(' ')[-1])
        scaler = StandardScaler(withMean=True, withStd=True).fit(features)
        self.trainSet = labels.zip(scaler.transform(features)).map(lambda _ : LabeledPoint(_[0], _[1]))


        testDirectory = HDFS_DIRECTORY + 'checkerboard4x4_test.txt'
        test = sc.textFile(testDirectory)
        features = test.map(lambda _ : _.split(' ')[:-1])
        labels = test.map(lambda _: _.split(' ')[-1])
        scaler = StandardScaler(withMean=True, withStd=True).fit(features)
        self.testSet = labels.zip(scaler.transform(features)).map(lambda _ : LabeledPoint(_[0], _[1]))






ds = DatasetCheckerboard2x2()






