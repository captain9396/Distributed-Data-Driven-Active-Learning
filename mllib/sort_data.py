import datetime
from datetime import timedelta
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors


# setup spark context and config
# conf = SparkConf().setAppName("test")

# conf = SparkConf().setAppName("Print Elements of RDD")\
#     .setMaster("local[4]").set("spark.executor.memory","1g");

sc = SparkContext.getOrCreate()
sc.setLogLevel("ERROR")


t1 = datetime.datetime.now()
r = sc.textFile('hdfs://node1:9000/input/small.txt', 20)
r = r.filter(lambda _ : _!='')
r = r.map(lambda  _ : int(_))
r = r.sortBy(lambda _ : _)
print('total samples is  = ' + str(r.count()))
print('no. of partitions = ' + str(r.getNumPartitions()))
print(r.collect())


t2 =  datetime.datetime.now()
time_difference = t2 - t1
time_difference_in_minutes = time_difference / timedelta(minutes=1)
print(str(time_difference_in_minutes * 60) + ' seconds !!')

