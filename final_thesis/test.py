from pyspark import  SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from sklearn.preprocessing import normalize
import numpy as np
# setup spark context and config
conf = SparkConf().setAppName("labeledPoints")
sc = SparkContext(conf=conf)
# sc.setLogLevel("ERROR")



data = sc.textFile('hdfs://node1:9000/input/data.txt')
data = data.map(lambda _ : np.array(_.strip().split()).astype(float))
data = data.map(lambda _ : _/np.linalg.norm(_))


print(data.take(1))





