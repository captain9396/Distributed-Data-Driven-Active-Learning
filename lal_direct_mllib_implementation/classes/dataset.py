from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.feature import StandardScaler, StandardScalerModel
import random
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
# fileDirectory = 'striatum_train.mini.txt'
# fileDirectory = 'striatum_test_mini.txt'
# fileDirectory = 'rotated_checkerboard2x2_test.txt'
# fileDirectory = 'rotated_checkerboard2x2_train.txt'




## here some changes needed to be made <<<<<<<<<<<


class Dataset:

    def __init__(self):
        # each dataset will have training and test data with label
        self.trainSet = None
        self.testSet = None


    def setStartState(self, nStart):
        '''
            Input:
            nStart -- number of labelled datapoints (size of indicesKnown)

        This functions initialises fields indicesKnown and
            indicesUnknown which contain the indices of labelled and
            unlabeled datapoints for example if nStart = 5, then there
            will be initially 5 labeled instances to start with.
            The basic workflow is first we get 1 +ve and 1 -ve label
            then we take the rest of the instances and shuffle them
            randomly finally we have TWO class members:

            self.indicesKnown = the 'nStart' number of labeled instances to start the lal-proceduce initially
            self.indicesUnknown = the rest of the UNlabeled instances


            N.B.
            By default nStart = 2, i.e. we will have 2 instances from each of the classes as initially labeled dataset
            in self.indicesKnown.
            But if nStart > 2, then we would first take 2 from the datasets and take another (nStart-2) instances randomly
            and add them to self.indicesKnown.
        '''

        self.nStart = nStart
        # indexing the whole dataset
        self.trainSet = self.trainSet.zipWithIndex().map(lambda _ : (_[1] , _[0]))
        self.testSet = self.testSet.zipWithIndex().map(lambda _: (_[1], _[0]))


        # first get 1 positive and 1 negative point so that both classes are represented and initial
        # classifer could be trained. Here zipWithIndex() method takes the labels one-by-one and forms
        # rdd of (label,index) pairs then we filter it to get the indices of positive labels only

        positiveIndices = self.trainSet\
            .filter(lambda _ : _[1].label == 1.0)\
            .map(lambda _ : _[0])

        # permuting the positive indices randomly
        shuffledPositiveIndices = positiveIndices.sortBy(lambda _: random.random())


        # sorting out the negative labels with their indices
        negativeIndices = self.trainSet\
            .filter(lambda _ : _[1].label == 0.0)\
            .map(lambda _ : _[0])
        # permuting the negative indices randomly
        shuffledNegativeIndices = negativeIndices.sortBy(lambda _: random.random())

        # taking one from each of the classes
        self.indicesKnown = sc.parallelize([shuffledPositiveIndices.take(1), shuffledNegativeIndices.take(1)]).map(lambda _ : _[0])


        # gathering all the rest of the labels together
        restOfThePositives = shuffledPositiveIndices.subtract(self.indicesKnown)
        restOfTheNegatives = shuffledNegativeIndices.subtract(self.indicesKnown)
        indicesRestAll = restOfThePositives.union(restOfTheNegatives)

        # permute them
        indicesRestAll = indicesRestAll\
            .sortBy(lambda _: random.random())

        # if we need more than 2 datapoints, select the rest nStart-2 at random
        if nStart > 2:
            # concatenating initially 2 known instances with another (nStart-2, so total = nStart) instances from first
            self.indicesKnown = self.indicesKnown\
                .union(indicesRestAll.zipWithIndex()
                       .filter(lambda _ : _[1] < nStart-2)
                       .map(lambda _ : _[0]))

        # the rest of the points will be unlabeled at the beginning
        # here we are taking all after first 'nStart' items
        self.indicesUnknown = indicesRestAll.zipWithIndex()\
            .filter(lambda _ : _[1] >= nStart-2)\
            .map(lambda _ : _[0])


        # print("########################## no. of labeled instances = " , self.indicesKnown.count())
        # print("########################## no. of UNlabeled instances = ", self.indicesUnknown.count())
        # print('------------------- LABELED INSTANCES -------------------\n' , self.indicesKnown.collect())
        # print('------------------- UNLABELED INSTANCES -------------------\n', self.indicesUnknown.collect())
        # print('------------------- Trainset -------------------\n', self.trainSet.take(10))
        # print('------------------- Testset -------------------\n', self.testSet.take(10))










class DatasetCheckerboard2x2(Dataset):
    '''Loads XOR-like dataset of checkerboard shape of size 2x2.
        Origine of the dataset: generated by Kseniya et al. '''

    def __init__(self):
        Dataset.__init__(self)


        # preparing the Data (Train and Test) : formatting and scaling then making it an RDD of LabeledPoints

        trainDirectory =  HDFS_DIRECTORY + 'checkerboard2x2_train.txt'
        train = sc.textFile(trainDirectory)
        features = train.map(lambda _ : _.split(' ')[:-1])
        labels = train.map(lambda _: _.split(' ')[-1])
        scaler = StandardScaler(withMean=True, withStd=True).fit(features)
        self.trainSet = labels.zip(scaler.transform(features))\
            .map(lambda _: LabeledPoint(_[0], _[1]))

        testDirectory = HDFS_DIRECTORY + 'checkerboard2x2_test.txt'
        test = sc.textFile(testDirectory)
        features = test.map(lambda _ : _.split(' ')[:-1])
        labels = test.map(lambda _: _.split(' ')[-1])
        scaler = StandardScaler(withMean=True, withStd=True).fit(features)
        self.testSet = labels.zip(scaler.transform(features))\
            .map(lambda _: LabeledPoint(_[0], _[1]))



        ''' this block is for testing '''
        # reg = RandomForest.trainRegressor(self.trainSet, numTrees=100, categoricalFeaturesInfo={})
        # predictions = reg.predict(self.testSet.map(lambda x: x.features))
        # labelsAndPredictions = self.testSet.map(lambda lp: lp.label).zip(predictions)
        # testMSE = labelsAndPredictions.map(lambda x: (x[0] - x[1]) ** 2).sum() / float(self.testSet.count())
        # print('Test Mean Squared Error = ' + str(testMSE))





class DatasetCheckerboard4x4(Dataset):
    '''Loads XOR-like dataset of checkerboard shape of size 4x4.
            Origine of the dataset: generated by Kseniya et al. '''

    def __init__(self):
        Dataset.__init__(self)

        trainDirectory =  HDFS_DIRECTORY + 'checkerboard4x4_train.txt'
        train = sc.textFile(trainDirectory)
        features = train.map(lambda _ : _.split(' ')[:-1])
        labels = train.map(lambda _: _.split(' ')[-1])
        scaler = StandardScaler(withMean=True, withStd=True).fit(features)
        self.trainSet = labels.zip(scaler.transform(features))\
            .map(lambda _ : LabeledPoint(_[0], _[1]))


        testDirectory = HDFS_DIRECTORY + 'checkerboard4x4_test.txt'
        test = sc.textFile(testDirectory)
        features = test.map(lambda _ : _.split(' ')[:-1])
        labels = test.map(lambda _: _.split(' ')[-1])
        scaler = StandardScaler(withMean=True, withStd=True).fit(features)
        self.testSet = labels.zip(scaler.transform(features))\
            .map(lambda _ : LabeledPoint(_[0], _[1]))






class DatasetRotatedCheckerboard2x2(Dataset):
    '''Loads XOR-like dataset of checkerboard shape of size 2x2 that is rotated by 45'.
    Origine of the dataset: generated by Kseniya et al. '''

    def __init__(self):
        Dataset.__init__(self)

        trainDirectory = HDFS_DIRECTORY + 'rotated_checkerboard2x2_train.txt'
        train = sc.textFile(trainDirectory)
        features = train.map(lambda _: _.split(' ')[:-1])
        labels = train.map(lambda _: _.split(' ')[-1])
        scaler = StandardScaler(withMean=True, withStd=True).fit(features)
        self.trainSet = labels.zip(scaler.transform(features)) \
            .map(lambda _: LabeledPoint(_[0], _[1]))

        testDirectory = HDFS_DIRECTORY + 'rotated_checkerboard2x2_test.txt'
        test = sc.textFile(testDirectory)
        features = test.map(lambda _: _.split(' ')[:-1])
        labels = test.map(lambda _: _.split(' ')[-1])
        scaler = StandardScaler(withMean=True, withStd=True).fit(features)
        self.testSet = labels.zip(scaler.transform(features)) \
            .map(lambda _: LabeledPoint(_[0], _[1]))






class DatasetStriatumMini(Dataset):
    '''Dataset from CVLab. https://cvlab.epfl.ch/data/em
    Features as in A. Lucchi, Y. Li, K. Smith, and P. Fua. Structured
     Image Segmentation Using Kernelized Features. ECCV, 2012'''

    def __init__(self):
        Dataset.__init__(self)

        trainDirectory = HDFS_DIRECTORY + 'striatum_train_mini.txt'
        train = sc.textFile(trainDirectory)
        features = train.map(lambda _: _.strip().split(' ')[:-1])
        labels = train.map(lambda _: _.strip().split(' ')[-1])
        scaler = StandardScaler(withMean=True, withStd=True).fit(features)
        self.trainSet = labels.zip(scaler.transform(features)) \
            .map(lambda _: LabeledPoint(0 if _[0] == '-1' else 1, _[1]))



        testDirectory = HDFS_DIRECTORY + 'striatum_test_mini.txt'
        test = sc.textFile(testDirectory)
        features = test.map(lambda _: _.split(' ')[:-1])
        labels = test.map(lambda _: _.split(' ')[-1])

        # AN ISSUE HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # in original LAL code they scaled testset with the scaler fitted from TRAINING set, but why?

        scaler = StandardScaler(withMean=True, withStd=True).fit(features)
        self.testSet = labels.zip(scaler.transform(features)) \
            .map(lambda _: LabeledPoint(0 if _[0] == '-1' else 1, _[1]))











# ds = DatasetRotatedCheckerboard2x2()

# ds.setStartState(7)


