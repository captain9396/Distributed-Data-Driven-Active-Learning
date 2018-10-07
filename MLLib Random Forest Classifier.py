from pyspark import  SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils





conf = SparkConf().setAppName("labeledPoints")
sc = SparkContext(conf=conf)




rdd = sc.textFile("file:///home/ubuntu/DATASETS/BIG_DATASETS/breast-cancer-wisconsin.data")
rdd = rdd.map(lambda _ : _.strip().split(','))

# filtering dataset for null values
rdd = rdd.filter(lambda _ : '?' not in _)


# labeling with 0 and 1 as there are only 2 classes. Default in this dataset was '2' and '4' 
# but using those values gives error when model starts to train
rdd = rdd.map(lambda _ : LabeledPoint(0 if _[-1]=='2' else 1, _[:-1]))


# splitting dataset
train, test = rdd.randomSplit([80.0, 20.0])

# building model
model = RandomForest.trainClassifier(train, numClasses=2, categoricalFeaturesInfo={},numTrees=100, featureSubsetStrategy="auto",impurity='gini')


# returns a list of predicted labels 
predictions = model.predict(test.map(lambda x: x.features))

# getting tuples of (testLabel, predictedLabel)
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)


# seeing which of the tuples are inequal in value
testErr = labelsAndPredictions.filter(lambda _ : _[0] != _[1])


# get error percentage
print('Test Error : ' , testErr.count() / test.count())



# get debugstring of the model
# print(model.toDebugString())


# model can't be saved !
# model.save(sc, '/home/ubuntu/SOURCE_CODE/src/myRandomForestClassifierModel')
# model.save(sc, 'file:///home/ubuntu/SOURCE_CODE/src/myRandomForestClassifierModel')
# model.save(sc, 'myRandomForestClassifierModel')