import pyspark
# As of Spark 2.0, the RDD-based APIs in the spark.mllib package have entered maintenance mode.
# The primary Machine Learning API for Spark is now the DataFrame-based API in the spark.ml package.

#Set up spark Context, for pyspark shell, no need.
conf = pyspark.SparkConf()
conf.setAppName('Interpolation')
conf.setMaster('local[*]')
sc = pyspark.SparkContext(conf=conf)

#Load sqlContext
spark = pyspark.sql.SparkSession.builder \
    .master("local[*]") \
    .appName("Interpolation") \
    .getOrCreate()

#DataFrames ->
#import pandas as pd
#T1_grid = pd.DataFrame([B1errmat.flatten(),model.flatten(),T1mat.flatten()],index=["B1err","ratio","T1"])
#T1_grid_trans = T1_grid.T # Transpose pandas dataframe

#pandas to sparksql
#T1_df = spark.createDataFrame(T1_grid_trans) #Takes longer the bigger the df
#T1_grid_trans.to_csv("T1_Lookup.csv")
T1_df2=spark.read.csv("T1_Lookup.csv", header = True, inferSchema=True) # faster than converting pandas DF to sparkDF

#knots of ratio and B1

from pyspark.sql import functions as F
#  T1_df2.agg({"B1err": "max"}).collect()[0][0]

knotRatio = np.linspace(0.00003,2.7,10)
knotB1 = np.linspace(0.1,2.0,10)
#################################OLD METHOD - save commands as list of string and use eval #############################
#knotCol = []
#for c in knotRatio:
#    knotCol.append("F.when(T1_df2.ratio > {}, T1_df2.ratio - {}).otherwise(0)".format(c,c))

#knotRatioCol = F.array([eval(knotCol[x]) for x in range(1,10)]).alias("RatioKnots") # One column containing all knots for Ratio
#T1_df3 = T1_df2.select('*', knotRatioCol) #This adds column

#knotCol = []
#for c in knotB1:
#    knotCol.append("F.when(T1_df2.B1err > {}, T1_df2.B1err - {}).otherwise(0)".format(c,c))

#knotB1Col = F.array([eval(knotCol[x]) for x in range(1,10)]).alias("B1Knots") # One column containing all knots for B1
#T1_df4 = T1_df3.select('*', knotB1Col) #This adds column
########################################################################################################################

#New cooler method: simply put the for loop commands into an array and select:
T1_df3 = T1_df2.select('*',F.array([F.when(T1_df2.ratio > c, T1_df2.ratio - c).otherwise(0) for c in knotRatio]).alias("RatioKnots"))
T1_df4 = T1_df3.select('*',F.array([F.when(T1_df2.B1err > c, T1_df2.B1err - c).otherwise(0) for c in knotB1]).alias("B1Knots"))

#Convert arrays to vectors: credit https://stackoverflow.com/questions/42138482/pyspark-how-do-i-convert-an-array-i-e-list-column-to-vector
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf

list_to_vec = udf(lambda x: Vectors.dense(x),VectorUDT())
df_with_vectors = T1_df4.select('B1err','ratio','T1', list_to_vec(T1_df4["B1Knots"]).alias("B1Knots"), list_to_vec(T1_df4["RatioKnots"]).alias("RatioKnots"))

###Regression Time!
from pyspark.ml.feature import VectorAssembler
vec = VectorAssembler(inputCols=["B1err","ratio","B1Knots","RatioKnots"],outputCol="features")
T1_df5 = vec.transform(df_with_vectors)

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(labelCol="T1",featuresCol="features")
model = lr.fit(T1_df5)

#Test model predictions:
#Trying to use same logic as above
x1=np.linspace(0.1,2,100) #B1err
x2=np.linspace(0.0005,2.5,100) #Ratio
x1_2 = np.zeros([100,100])
x2_2 = np.zeros([100,100])
for i in range(0,len(x1)):
    for j in range(0,len(x2)):
        x1_2[i, j] = x1[i]
        x2_2[i, j] = x2[j]

x1 = x1_2.flatten()
x2 = x2_2.flatten()

a_df = pd.DataFrame([x1,x2], index=["B1err","ratio"])
a_df = a_df.T #2 rows of 10000 (100 B1 * 100 ratio values)

#Feature vectors have to be piecewise: knotB1 knotRatio keep in my mind order
a_df2 = spark.createDataFrame(a_df)

a_df3 = a_df2.select('*',F.array([F.when(a_df2.B1err > c, a_df2.B1err - c).otherwise(0) for c in knotB1]).alias("B1Knots"))
a_df4 = a_df3.select('*',F.array([F.when(a_df2.ratio > c, a_df2.ratio - c).otherwise(0) for c in knotRatio]).alias("RatioKnots"))

#Using our wonderful list_to_vec
a_df5 = a_df4.select('B1err','ratio', list_to_vec(a_df4["B1Knots"]).alias("B1Knots"), list_to_vec(a_df4["RatioKnots"]).alias("RatioKnots"))
vec = VectorAssembler(inputCols=["B1err","ratio","B1Knots","RatioKnots"],outputCol="features")
a_df6 = vec.transform(a_df5)
result = model.transform(a_df6)

#Back to pandas for plotting:
dd = result.toPandas()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
#Plot orignal data as surface:
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(dd["B1err"].values,dd["ratio"].values,dd["prediction"].values, cmap=plt.cm.jet, linewidth=0.2, antialiased=True)
ax.set_xlabel('B1')
ax.set_ylabel('Ratio')
ax.set_zlabel('T1')
ax.view_init(azim=210)
plt.show()

from pyspark.ml.feature import PolynomialExpansion
#Todo Clustering?
