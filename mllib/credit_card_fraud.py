from pyspark import  SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
import datetime
from datetime import timedelta




# setup spark context and config
conf = SparkConf().setAppName("labeledPoints")
sc = SparkContext(conf=conf)

# get starting time
t1 = datetime.datetime.now()

# create an RDD
data = sc.textFile('file:///home/ubuntu/DATASETS/BIG_DATASETS/creditcard.csv')

# preprocess data
data = data.filter(lambda _  : _[0][0] != '"')
data = data.map(lambda _ : _.split(','))
data = data.map(lambda row : LabeledPoint(float(row[-1][1]), row[:-1]))


# split data into train-test set
train, test = data.randomSplit([70.0, 30.0])

# if needed, feel free to release memory
# data.unpersist()


# training model
model = RandomForest.trainClassifier(train, numClasses=2, \
	categoricalFeaturesInfo={},numTrees=100, featureSubsetStrategy="auto",impurity='gini')



# calculate time needed
t2 = datetime.datetime.now()
time_difference = t2 - t1
time_difference_in_minutes = time_difference / timedelta(minutes=1)
print('Time elapsed = ' , time_difference_in_minutes , ' minutes')




# getting class predictions
predictions = model.predict(test.map(lambda x: x.features))

# creating RDD of pairs of (true_label, predicted_label)
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)

# calculate total wrongly predicted pairs in the RDD
testErr = labelsAndPredictions.filter(lambda _ : _[0] != _[1])

# accuracy score
print('Test Accuracy : ' , 1 - (testErr.count() / test.count()))





