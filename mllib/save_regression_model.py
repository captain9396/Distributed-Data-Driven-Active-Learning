from pyspark import  SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
import datetime
from datetime import timedelta



# setup spark context and config
conf = SparkConf().setAppName("labeledPoints")
sc = SparkContext(conf=conf)



t1 = datetime.datetime.now()


data = sc.textFile('hdfs://node1:9000/input/checkerboard2x2_train.txt')
data = data.map(lambda _ : _.split(' '))
data = data.map(lambda row : LabeledPoint(row[-1], row[:-1]))



train, test = data.randomSplit([70.0, 30.0])

# training model
try:
    reg = RandomForestModel.load(sc, 'hdfs://node1:9000/input/model')
    print('Saved Model Loaded from HDFS !')
except:
    reg = RandomForest.trainRegressor(train, numTrees=100, categoricalFeaturesInfo={})
    reg.save(sc, 'hdfs://node1:9000/input/model')
    print('Model Trained')

t2 = datetime.datetime.now()
time_difference = t2 - t1
time_difference_in_minutes = time_difference / timedelta(minutes=1)
print('Time elapsed = ' , time_difference_in_minutes , ' minutes')




predictions = reg.predict(test.map(lambda x: x.features))



# creating RDD of pairs of (true_label, predicted_label)
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)

testMSE = labelsAndPredictions.map(lambda x : (x[0] - x[1]) ** 2).sum() / float(test.count())

print('Test Mean Squared Error = ' + str(testMSE))






