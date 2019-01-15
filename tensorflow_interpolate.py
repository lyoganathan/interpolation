# Originall trying to do universal function approximation of this 2D function:
# http://mcneela.github.io/machine_learning/2017/03/21/Universal-Approximation-Theorem.html
# But couldn't get a good result with just one layer so doing 2.
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

T1_Lookup = pd.read_csv("T1_Lookup.csv", usecols=["B1err","ratio","T1"])
T1_Lookup = pd.read_csv("T1_Lookup_Table_Lowres.csv", usecols=["B1err","ratio","T1"])

#Max-min normalize or z score normalize?? MinMax keeps between 0 and 1
#scl_x = MinMaxScaler()
scl_y = MinMaxScaler()
scl_x = MinMaxScaler()

#a = scl.fit_transform(T1_Lookup)
y_train = scl_y.fit_transform(T1_Lookup[["T1"]].values.reshape(-1,1))
X_train = scl_x.fit_transform(T1_Lookup[["B1err","ratio"]].values.reshape(-1,2))

# Look at what Scaled data looks like to make sure it's the same
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(X_train[:,0],X_train[:,1],y_train[:,0], cmap=plt.cm.jet, linewidth=0.2, antialiased=True)
#
# ax.set_xlabel('B1')
# ax.set_ylabel('Ratio')
# ax.set_zlabel('T1')
# ax.view_init(azim=210)
# plt.show()

#Prediction Data:
x1=np.linspace(0.1,2,100)
x2=np.linspace(0.0005,2.5,100)
x1_2 = np.zeros([100,100])
x2_2 = np.zeros([100,100])
for i in range(0,len(x1)):
    for j in range(0,len(x2)):
        x1_2[i, j] = x1[i]
        x2_2[i, j] = x2[j]

b = np.zeros([len(x1_2.flatten()),2])
b[:,0] = x1_2.flatten()
b[:,1] = x2_2.flatten()

X_test = scl_x.transform(b)

#X_test= b
# # CNN
# # Conv1d:
# #Input shape: 3D tensor with shape: (batch_size, steps, input_dim)
# #Output shape: 3D tensor with shape: (batch_size, new_steps, filters) steps value might have changed due to padding or strides.
#
# # trying conv2d
# input_dim = len(T1_Lookup) # Entire data...
# batch_size = 1
# patch_size = 5
# depth = 16 # Just choes this randomly...
# num_hidden1 = 256
# hidden_dim = 64
# output_dim= input_dim
# num_channels = 2 # bivariate distribution, is this the same as doing width = 2 or height = 2 instead?
#
# # https://www.quora.com/What-do-channels-refer-to-in-a-convolutional-neural-network
#
# # xs: batch, height, width, channel
# xs = tf.placeholder("float", shape=(batch_size, 1, input_dim, num_channels)) #input dim is window size? -> do whole function
# ys = tf.placeholder("float", shape=(batch_size,output_dim))
#
# # w: height, width, in_channels, out channels
# initializer = tf.contrib.layers.xavier_initializer()
# W_1 = tf.Variable(initializer([1,1,num_channels,depth])) # depth is random, think you can play with filter size?
# b_1 = tf.Variable(initializer([depth]))
#
# x = tf.nn.conv2d(xs,W_1,strides=[1,1,1,1],padding="SAME")
# x = tf.nn.bias_add(x,b_1)
# layer_1 = tf.nn.relu(x)
# pool_1 = tf.nn.max_pool(layer_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") #Do you need pool layer?
#
# # Flattern the convolution output
# shape = pool_1.get_shape().as_list()
# reshape = tf.reshape(pool_1, [shape[0], shape[1] * shape[2] * shape[3]])
#
# W_O = tf.Variable(initializer([hidden_dim,output_dim]))
# b_O = tf.Variable(tf.zeros([output_dim]))
#
# # FC hidden layers
# output = tf.add(tf.matmul(reshape, W_O) + b_O)
#
# # Minimize output prediction minus label # our mean squared error cost function
# cost = tf.reduce_mean(tf.square(output-ys)) # y-y_true #This is where ys goes, output is calculated using xs
# #cost = tf.nn.l1_loss(output-ys)
# cost_summary_t = tf.summary.scalar('loss', cost)
# # Gradinent Descent optimiztion
# train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
# #train = tf.train.AdamOptimizer(0.0001).minimize(cost) # Adam optimizer



# Dense Neural net
# This where we pass our input data? X is features y is labels?

xs = tf.placeholder("float", [None,2])
ys = tf.placeholder("float", [None,1])
input_dim=2
output_dim=1
hidden_dim=100
hidden_dim_2=100

#Random initialization:
# W_1 = tf.Variable(tf.random_uniform([input_dim,hidden_dim]))
# b_1 = tf.Variable(tf.zeros([hidden_dim]))
#Xavier method:
initializer = tf.contrib.layers.xavier_initializer()
#Also xavier method:
#initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True) <- equivilant?

#Weightes and biases that change each training step:
W_1 = tf.Variable(initializer([input_dim,hidden_dim])) # xavier initialize weights
b_1 = tf.Variable(tf.zeros([hidden_dim])) # initialize bias with 0

# layer 1 multiplying and adding bias, then applying activation function
layer_1 = tf.add(tf.matmul(xs,W_1), b_1) #This is where xs goes
layer_1 = tf.nn.relu(layer_1)

W_2 = tf.Variable(initializer([hidden_dim,hidden_dim_2]))
b_2 = tf.Variable(tf.zeros([hidden_dim_2]))

layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
layer_2 = tf.nn.softmax(layer_2)

#output layer has one node only since performing regression
W_O = tf.Variable(initializer([hidden_dim_2,output_dim]))
b_O = tf.Variable(tf.zeros([output_dim]))
output = tf.add(tf.matmul(layer_2,W_O), b_O) # This our neural net?

# Minimize output prediction minus labels, our mean squared error cost function (MSE)
cost = tf.reduce_mean(tf.square(output-ys) + 0.01*tf.nn.l2_loss(W_O)) # y-y_true #This is where ys goes, output is calculated using xs
#Apply l2 regularization to W1 and W0:
#cost =tf.reduce_mean(tf.square(output-ys) + 0.01*tf.nn.l2_loss(W_1) + 0.01*tf.nn.l2_loss(W_O))
#Log loss approches:
#cost = tf.losses.log_loss(ys,output)
#cost = tf.reduce_sum(tf.multiply(-ys, tf.log(output)))
cost_summary_t = tf.summary.scalar('loss', cost)
# Gradinent Descent optimiztion
#train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
train = tf.train.AdamOptimizer(0.05).minimize(cost) # Adam optimizer

with tf.Session() as sess:
    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sess,'yahoo_dataset.ckpt')
    for i in range(5000):
        #Select 10000 random uniform points from function, out of the 144690
        idx = np.random.randint(0, len(X_train), [10000, 1])
        #idx=np.arange(0,len(X_train),1) # Select entire dataset - slow but works as well
        x_in = X_train[np.sort(idx.flatten())]
        y_in = y_train[np.sort(idx.flatten())]

        # Run cost and train with each sample
        current_loss, loss_summary, _ = sess.run([cost,cost_summary_t,train],feed_dict={xs: x_in, ys: y_in})
        #Feed dict changes with each iteration, so does train? and cost? we pass ys a placeholder with y_train, same with xs.

        if (i + 1) % 10 == 0:
            print('batch: %d, loss: %f' % (i + 1, current_loss))

    # predict output of test data after training
    pred = sess.run(output, feed_dict={xs: X_test})

    #This evaluates our neural net at the 100x100 grid we set up earlier
    x_result = scl_x.inverse_transform(X_test)
    y_result = scl_y.inverse_transform(pred)
    result = np.c_[x_result, y_result]
    #saver.save(sess, r'C:\Users\laagi_000\Documents\Laagi\model') # Save model

#Plot the result:
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(result[:,0],result[:,1],result[:,2], cmap=plt.cm.jet, linewidth=0.2, antialiased=True)
# ax.set_xlim(0.1,2)
# ax.set_ylim(0,3)
# ax.set_zlim(0,4)
ax.set_xlabel('B1')
ax.set_ylabel('Ratio')
ax.set_zlabel('T1')
ax.view_init(azim=210)
plt.show()

