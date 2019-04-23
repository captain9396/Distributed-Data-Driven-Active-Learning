from pyspark.accumulators import AccumulatorParam


class VectorAccumulatorParam(AccumulatorParam):
     def zero(self, value):
         return [0.0] * len(value)

     def addInPlace(self, val1, val2):
         for i in range(len(val1)):
              val1[i] += val2[i]
         return val1