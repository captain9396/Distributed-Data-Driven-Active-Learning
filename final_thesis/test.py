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


def to_matrix_entry(s):
        ss = s.split()
	entry =	MatrixEntry(float(ss[0]), float(ss[1]), float(ss[2]))
	return entry
	



data = sc.textFile('hdfs://node1:9000/input/sqr.txt')
mat = data.map(to_matrix_entry)
rdd = sc.parallelize(mat.collect())


coord_mat = CoordinateMatrix(rdd)
coord_mat = coord_mat.transpose()
row_mat = coord_mat.toRowMatrix()
sim = row_mat.columnSimilarities()
print(sim.entries.take(10))


debug.TIMESTAMP(2)











'''
data = data.map(lambda _ : np.array(_.strip().split()).astype(float))
unitMatrix = data.map(lambda _ : _/np.linalg.norm(_))

#unitMatrix = sc.parallelize(np.array([[1,2,3,5,7], [6,2,1,-1,3], [7,0,1,2,-4]]).T)
mat = RowMatrix(unitMatrix)
S = mat.columnSimilarities()

sims = S.entries.collect()
print(len(sims))
print(sims)


SrowMatrix = S.toRowMatrix()
print(SrowMatrix.rows.map(lambda _ : DenseVector(_.toArray())).collect())

'''
