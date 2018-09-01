#Piecewise Linear regression in sklearn:
from sklearn import linear_model
import pandas as pd

T1_Lookup = pd.read_csv("T1_Lookup.csv", usecols=["B1err","ratio","T1"])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
#Plot orignal data as surface:
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(T1_Lookup["B1err"],T1_Lookup["ratio"],T1_Lookup["T1"], cmap=plt.cm.jet, linewidth=0.2, antialiased=True)
ax.set_xlabel('B1')
ax.set_ylabel('Ratio')
ax.set_zlabel('T1')
ax.view_init(azim=210)
plt.show()

#Piecewise function: http://songhuiming.github.io/pages/2015/09/22/piecewise-linear-function-and-the-explanation/
#Legendary stack post: https://stackoverflow.com/questions/37872171/how-can-i-perform-two-dimensional-interpolation-using-scipy

#Todo Clustering?

#Simple cubic surface fit:

#Build model:
x1=T1_Lookup["B1err"]
x2=T1_Lookup["ratio"]
# y = x1 + x1^2 + x1^3 + x2 + x2^2 + x2^3 (you can optionally include intercept terms)
model_eq = [x1,np.power(x1,2),np.power(x1,3),x2,np.power(x2,2),np.power(x2,3)]
#model_eq = [x1,x2] # for ridge regression
model_df = pd.DataFrame(model_eq)
model_df = model_df.T
lm = linear_model.LinearRegression()
#lm = linear_model.Lasso(alpha=0.1)
#lm = linear_model.RANSACRegressor()
#from sklearn import kernel_ridge
#lm = kernel_ridge.KernelRidge(alpha=0.1,degree=1)
lm.fit(model_df,T1_Lookup["T1"])

#Data to interpolate - a regular 100X100 grid of T1 and B1 values
x1=np.linspace(0.1,2,100)
x2=np.linspace(0.0005,2.5,100)
x1_2 = np.zeros([100,100])
x2_2 = np.zeros([100,100])
for i in range(0,len(x1)):
    for j in range(0,len(x2)):
        x1_2[i, j] = x1[i]
        x2_2[i, j] = x2[j]

x1 = x1_2.flatten()
x2 = x2_2.flatten()
#Data to evaluate also has to follow our linear model equations, meaning we have to supply all x values in the equation
predict_eq=[x1,np.power(x1,2),np.power(x1,3),x2,np.power(x2,2),np.power(x2,3)]
predict_df = pd.DataFrame(predict_eq)
predict_df = predict_df.T
x3=lm.predict(predict_df)

#Plot prdictions/interpolations:
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x1, x2, x3.flatten(), cmap=plt.cm.jet, linewidth=0.2, antialiased=True)
ax.set_xlabel('B1')
ax.set_ylabel('Ratio')
ax.set_zlabel('T1')
ax.view_init(azim=210)
plt.show()

#Piecewise function set-up, using 10 knots:
B1_len=len(T1_Lookup["B1err"])
knotRatio = np.linspace(0.005,2.5,100)
knotB1 = np.linspace(0.2,2,100)

totalRatio = np.zeros([B1_len,len(knotRatio)+1])
totalRatio[:,0]=T1_Lookup["ratio"]
totalB1 = np.zeros([B1_len,len(knotB1)+1])
totalB1[:,0]=T1_Lookup["B1err"]

vecRatio = T1_Lookup["ratio"]
vecB1 = T1_Lookup["B1err"]
for i in range (0,len(knotRatio)): #last i is 10
    a = np.where(vecRatio > knotRatio[i], vecRatio - knotRatio[i], 0)
    totalRatio[:,i+1] = a
for i in range (0,len(knotB1)): #last i is 10
    a = np.where(vecB1 > knotB1[i], vecB1 - knotB1[i], 0)
    totalB1[:,i+1] = a

#Quadratic Surface, (response surface?): http://home.iitk.ac.in/~shalab/regression/Chapter12-Regression-PolynomialRegression.pdf
x1=totalB1
x2=totalRatio
#total = [x1,np.power(x1,2),x2,np.power(x2,2), x1 * x2]
#total = np.concatenate([x1,np.power(x1,2),x2,np.power(x2,2), x1 * x2],axis=1) #qudratic piecewise
total = np.concatenate([x1,x2,x1*x2],axis=1) # linear piecewise
ratio_df = pd.DataFrame(total)
#ratio_df = ratio_df.T
T1_df = pd.DataFrame(T1_Lookup["T1"])

lm = linear_model.LinearRegression()
lm.fit(ratio_df,T1_df) # <- exactly what spark linear regression does

#Plotting sklearn linear models

#Plot predictions:

#Set up values to be predicted: (num of features have to match)

# Features also have to be 'piecewise'
#zeros of length
x1=np.linspace(0.1,2,100)
x2=np.linspace(0.0005,2.5,100)
x1_2 = np.zeros([100,100])
x2_2 = np.zeros([100,100])
for i in range(0,len(x1)):
    for j in range(0,len(x2)):
        x1_2[i, j] = x1[i]
        x2_2[i, j] = x2[j]

x1 = x1_2.flatten()
x2 = x2_2.flatten()

B1_len=len(x1)
x1_2 = np.zeros([B1_len,len(knotB1)+1])
x1_2[:,0]=x1
x2_2 = np.zeros([B1_len,len(knotRatio)+1])
x2_2[:,0]=x2

vecRatio = x2
vecB1 = x1

for i in range (0,len(knotRatio)): #last i is 10
    a = np.where(vecRatio > knotRatio[i], vecRatio - knotRatio[i], 0)
    x2_2[:,i+1] = a
for i in range (0,len(knotB1)): #last i is 10
    a = np.where(vecB1 > knotB1[i], vecB1 - knotB1[i], 0)
    x1_2[:,i+1] = a

b=np.concatenate([x1_2,x2_2,x1_2*x2_2],axis=1)

a_df = pd.DataFrame(b)
x3=lm.predict(a_df)


#Plot data
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x1.flatten(), x2.flatten(), x3, s=0.5)
ax.set_xlabel('B1')
ax.set_ylabel('Ratio')
ax.set_zlabel('T1')
ax.view_init(azim=210)
plt.show()