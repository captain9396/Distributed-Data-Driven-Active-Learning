from pyspark.sql import SparkSession
from debugger import Debugger
from pyspark import  SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
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





data = sc.textFile('hdfs://node1:9000/input/vectors_50000x1000.txt')
data = data.map(lambda _ : np.array(_.strip().split()).astype(float))
data = data.map(lambda _ : _/np.linalg.norm(_))
U = data.zipWithIndex().map(lambda _ : IndexedRow(_[1], _[0]))
U = IndexedRowMatrix(U)



UT = U.toCoordinateMatrix()
UT = UT.transpose()



U = U.toBlockMatrix()
UT = UT.toBlockMatrix()

S = U.multiply(UT)

S_coord = S.toCoordinateMatrix()
sim = S_coord.entries
print(sim.take(100))


debug.TIMESTAMP(2)
