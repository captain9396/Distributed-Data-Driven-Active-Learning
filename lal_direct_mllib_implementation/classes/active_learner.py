from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.feature import StandardScaler, StandardScalerModel
from pyspark.mllib.stat import Statistics
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.tree import DecisionTreeModel
from dataset import *
from debugger import *
import math
import random
import datetime
from datetime import timedelta



# setup spark context and config
# conf = SparkConf().setAppName("test")

# conf = SparkConf().setAppName("Print Elements of RDD")\
#     .setMaster("local[4]").set("spark.executor.memory","1g");

sc = SparkContext.getOrCreate()
sc.setLogLevel("ERROR")

myDebugger = Debugger()



class ActiveLearner:
    '''This is the base class for active learning models'''

    def __init__(self, dataset, nEstimators, name):
        '''input: dataset -- an object of class Dataset or any inheriting classes
                  nEstimators -- the number of estimators for the base classifier, usually set to 50
                  name -- name of the method for saving the results later'''

        self.dataset = dataset
        self.indicesKnown = dataset.indicesKnown
        self.indicesUnknown = dataset.indicesUnknown
        # base classification model
        self.nEstimators = nEstimators
        self.model = None
        self.name = name


    def reset(self):
        '''forget all the points sampled by active learning and set labelled
         and unlabelled sets to default of the dataset'''
        self.indicesKnown = self.dataset.indicesKnown
        self.indicesUnknown = self.dataset.indicesUnknown




    def train(self):
        '''train the base classification model on currently available datapoints'''

        # first fetch the subset of training data which match the indices of the known indices
        self.trainDataKnown = self.indicesKnown.map(lambda _ : (_, None))\
            .leftOuterJoin(self.dataset.trainSet)\
            .map(lambda _ : (_[0], _[1][1]))

        # testData = self.dataset.testSet.map(lambda _ : _[1].features)


        # train a RFclassifer with this data
        self.model = RandomForest.trainClassifier(self.trainDataKnown.map(lambda _ : _[1]),
                                                  numClasses=2,
                                                  categoricalFeaturesInfo={},
                                                  numTrees=self.nEstimators,
                                                  featureSubsetStrategy="auto",
                                                  impurity='gini')




        # treeRdd = DecisionTreeModel(self.model._java_model.trees()[0])
        # myDebugger.DEBUG(treeRdd.predict(testData).collect())

        # trees = [DecisionTreeModel(self.model._java_model.trees()[i]) for i in range(100)]
        #
        #
        # predictions = [t.predict(testData) for t in trees]
        #
        # for pred in predictions:
        #     myDebugger.DEBUG(pred.count())


    # def evaluate(self, performanceMeasures):
    #
    #     '''evaluate the performance of current classification for a given set of performance measures
    #     input: performanceMeasures -- a list of performance measure that we would like to estimate. Possible values are 'accuracy', 'TN', 'TP', 'FN', 'FP', 'auc'
    #     output: performance -- a dictionary with performanceMeasures as keys and values consisting of lists with values of performace measure at all iterations of the algorithm'''
    #     performance = {}
    #     test_prediction = self.model.predict(self.dataset.testData)
    #     m = metrics.confusion_matrix(self.dataset.testLabels, test_prediction)
    #
    #     if 'accuracy' in performanceMeasures:
    #         performance['accuracy'] = metrics.accuracy_score(self.dataset.testLabels, test_prediction)
    #
    #     if 'TN' in performanceMeasures:
    #         performance['TN'] = m[0, 0]
    #     if 'FN' in performanceMeasures:
    #         performance['FN'] = m[1, 0]
    #     if 'TP' in performanceMeasures:
    #         performance['TP'] = m[1, 1]
    #     if 'FP' in performanceMeasures:
    #         performance['FP'] = m[0, 1]
    #
    #     if 'auc' in performanceMeasures:
    #         test_prediction = self.model.predict_proba(self.dataset.testData)
    #         test_prediction = test_prediction[:, 1]
    #         performance['auc'] = metrics.roc_auc_score(self.dataset.testLabels, test_prediction)
    #
    #     return performance





class DistributedActiveLearnerRandom(ActiveLearner):
    '''Randomly samples the points'''

    def selectNext(self):
        # permuting unlabeled instances randomly
        self.indicesUnknown = self.indicesUnknown\
            .sortBy(lambda _: random.random())

        # takes the first from the unknown samples and add it to the known ones
        self.indicesKnown = self.indicesKnown\
            .union(sc.parallelize(self.indicesUnknown.take(1)))

        # removing first sample from unlabeled ones(update)
        first = self.indicesUnknown.first()
        self.indicesUnknown = self.indicesUnknown.filter(lambda _ : _ != first)








class DistributedActiveLearnerUncertainty(ActiveLearner):
    '''Points are sampled according to uncertainty sampling criterion'''

    def selectNext(self):
        # predict for the rest the datapoints
        self.trainDataUnknown = self.indicesUnknown.map(lambda _: (_, None)) \
            .leftOuterJoin(self.dataset.trainSet) \
            .map(lambda _: (_[0], _[1][1]))

        actualIndices = self.trainDataUnknown.map(lambda _ : _[0])\
            .zipWithIndex()\
            .map(lambda _: (_[1], _[0]))

        myDebugger.TIMESTAMP('zipping indices ')


        rdd = sc.parallelize([])

        ''' these java objects are not serializable
         thus still no support to make an RDD out of it!! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        '''
        for x in self.model._java_model.trees():
            '''
             zipping each prediction from each decision tree
             with individual sample index so that they can be
             added later
            '''
            predX = DecisionTreeModel(x)\
                .predict(self.trainDataUnknown.map(lambda _ : _[1].features))\
                .zipWithIndex()\
                .map(lambda _: (_[1], _[0]))

            predX = actualIndices.leftOuterJoin(predX).map(lambda _ : _[1])
            rdd = rdd.union(predX)

        myDebugger.TIMESTAMP('get individual tree predictions')

        ''' adding up no. of 1 in each sample's prediction this is the class prediction of 1s'''
        classPrediction = rdd.groupByKey().mapValues(sum)

        myDebugger.TIMESTAMP('reducing ')


        #  direct self.nEstimators gives error
        totalEstimators = self.nEstimators
        #  predicted probability of class 0
        classPrediction = classPrediction.map(lambda _  : (_[0], abs(0.5 - (1-(_[1]/totalEstimators)))))

        myDebugger.TIMESTAMP('mapping')


        # Selecting the index which has the highest uncertainty/ closest to probability 0.5
        selectedIndex1toN = classPrediction.sortBy(lambda _ : _[1]).first()[0]


        myDebugger.TIMESTAMP('sorting')

        # takes the selectedIndex from the unknown samples and add it to the known ones
        self.indicesKnown = self.indicesKnown .union(sc.parallelize([selectedIndex1toN]))


        myDebugger.TIMESTAMP('update known indices')

        # removing first sample from unlabeled ones(update)
        self.indicesUnknown = self.indicesUnknown.filter(lambda _: _ != selectedIndex1toN)

        myDebugger.TIMESTAMP('update unknown indices')


        myDebugger.DEBUG(selectedIndex1toN)
        myDebugger.DEBUG(self.indicesKnown.collect())
        myDebugger.DEBUG(self.indicesUnknown.collect())


        myDebugger.TIMESTAMP('DEBUGGING DONE')






def getSD( x, totalEstimators):
    sumValue = x[1]
    mean = sumValue / totalEstimators
    sd = math.sqrt((sumValue * ((1 - mean) ** 2) + (totalEstimators - sumValue) * (mean ** 2)) / (totalEstimators-1))
    return (x[0], sd)



class ActiveLearnerLAL(ActiveLearner):
    '''Points are sampled according to a method described in K. Konyushkova, R. Sznitman, P. Fua 'Learning Active Learning from data'  '''

    def __init__(self, dataset, nEstimators, name, lalModel):
        ActiveLearner.__init__(self, dataset, nEstimators, name)
        self.lalModel = lalModel







    def selectNext(self):
        # get predictions from individual trees
        self.trainDataUnknown = self.indicesUnknown.map(lambda _: (_, None)) \
            .leftOuterJoin(self.dataset.trainSet) \
            .map(lambda _: (_[0], _[1][1]))

        actualIndices = self.trainDataUnknown.map(lambda _: _[0]) \
            .zipWithIndex() \
            .map(lambda _: (_[1], _[0]))

        rdd = sc.parallelize([])

        ''' these java objects are not serializable
         thus still no support to make an RDD out of it!! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        '''
        for x in self.model._java_model.trees():
            '''
             zipping each prediction from each decision tree
             with individual sample index so that they can be
             added later
            '''
            predX = DecisionTreeModel(x) \
                .predict(self.trainDataUnknown.map(lambda _: _[1].features)) \
                .zipWithIndex() \
                .map(lambda _: (_[1], _[0]))

            predX = actualIndices.leftOuterJoin(predX).map(lambda _: _[1])
            rdd = rdd.union(predX)

        ''' adding up no. of 1 in each sample's prediction this is the class prediction of 1s'''
        sumScore = rdd.groupByKey().mapValues(sum)
        totalEstimators = self.nEstimators


        # average of the predicted scores
        f_1 = sumScore.map(lambda _: (_[0], _[1] / totalEstimators))

        # standard deviation of predicted scores
        f_2 = sumScore.map(lambda _ : getSD(_,totalEstimators))
        









        # - average and standard deviation of the predicted scores
        # f_1 = np.mean(temp, axis=0)
        # f_2 = np.std(temp, axis=0)
        # # - proportion of positive points
        # f_3 = (sum(known_labels > 0) / n_lablled) * np.ones_like(f_1)
        # # the score estimated on out of bag estimate
        # f_4 = self.model.oob_score_ * np.ones_like(f_1)
        # # - coeficient of variance of feature importance
        # f_5 = np.std(self.model.feature_importances_ / n_dim) * np.ones_like(f_1)
        # # - estimate variance of forest by looking at avergae of variance of some predictions
        # f_6 = np.mean(f_2, axis=0) * np.ones_like(f_1)
        # # - compute the average depth of the trees in the forest
        # f_7 = np.mean(np.array([tree.tree_.max_depth for tree in self.model.estimators_])) * np.ones_like(f_1)
        # # - number of already labelled datapoints
        # f_8 = np.size(self.indicesKnown) * np.ones_like(f_1)
        #
        # # all the featrues put together for regressor
        # LALfeatures = np.concatenate(([f_1], [f_2], [f_3], [f_4], [f_5], [f_6], [f_7], [f_8]), axis=0)
        # LALfeatures = np.transpose(LALfeatures)
        #
        # # predict the expercted reduction in the error by adding the point
        # LALprediction = self.lalModel.predict(LALfeatures)
        # # select the datapoint with the biggest reduction in the error
        # selectedIndex1toN = np.argmax(LALprediction)
        # # retrieve the real index of the selected datapoint
        # selectedIndex = self.indicesUnknown[selectedIndex1toN]
        #
        # self.indicesKnown = np.concatenate(([self.indicesKnown, np.array([selectedIndex])]))
        # self.indicesUnknown = np.delete(self.indicesUnknown, selectedIndex1toN)


dtst = DatasetCheckerboard2x2()
# set the starting point
dtst.setStartState(2)

# alU = DistributedActiveLearnerUncertainty(dtst, 50, 'uncertainty')
#
# alU.train()
# myDebugger.TIMESTAMP('MODEL TRAINED!!')
# alU.selectNext()


alLALindepend = ActiveLearnerLAL(dtst, 50, 'lal-rand', '')
alLALindepend.train()
myDebugger.TIMESTAMP('MODEL TRAINED!!')
alLALindepend.selectNext()


