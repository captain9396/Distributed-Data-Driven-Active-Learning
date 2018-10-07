from pyspark import  SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils





conf = SparkConf().setAppName("labeledPoints")
sc = SparkContext(conf=conf)

row = ['2596,51,3,258,0,510']
rowRDD = sc.parallelize(row)


# A labeled point is an object which consists of a label(l) and a Vector(v) --> LabeledPoint(l,v)



#creating a labeled point RDD
labeledPointsRDD = rowRDD.map(lambda _ : _.split(',')).map(lambda _ : LabeledPoint(_[-1], _[:-2]))
labeledPointsRDD.collect()




#creating an RDD containing multiple Labeled Points

lp1 = LabeledPoint(1.0 , [1.0, 2.0, 3.0])
lp2 = LabeledPoint(1.0 , [1.0, 4.0, 7.0])
lp3 = LabeledPoint(0.0 , [8.0, 2.0, 0.0])
lp4 = LabeledPoint(1.0 , [4.0, 7.0, 1.0])

data = sc.parallelize([lp1, lp2, lp3, lp4])







# splitting the dataset into train and test set
# both are individual LabeledPoint RDDs 
train, test = data.randomSplit([80.0, 20.0])


model = RandomForest.trainClassifier(train, numClasses=2, categoricalFeaturesInfo={},numTrees=3, featureSubsetStrategy="auto",impurity='gini', maxDepth=4, maxBins=32)


