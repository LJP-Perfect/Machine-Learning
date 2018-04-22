from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


#import data
boston=load_boston()
data=boston.data[:,(5)]
label=boston.target

#reshape column vector
data=data.reshape(-1,1)
label=label.reshape(-1,1)

#feature scaling
data_scaler = preprocessing.MinMaxScaler()
target_scaler = preprocessing.MinMaxScaler()
data = data_scaler.fit_transform(data)
label = target_scaler.fit_transform(label)

#getting number of training set & number of feature
m_feature=data.shape[1]
m_train=tf.placeholder(dtype=tf.float32)

x=tf.placeholder(dtype=tf.float32,shape=[None,m_feature],name='input')
y_=tf.placeholder(dtype=tf.float32,shape=[None,1],name='output')

#Define weights and offsets viriables
#Initialize weights using a Gaussian distribution
#Initialize the bias to 0
w=tf.Variable(tf.truncated_normal(shape=[m_feature,1],stddev=0.003,dtype=tf.float32))
b=tf.Variable(tf.zeros(shape=[1],dtype=tf.float32))

#Forward calculation
y=tf.add(tf.matmul(x,w),b)

#use the mean square error function
learn_rate=0.08
loss=tf.reduce_sum(tf.pow(y_-y,2)/(2*m_train))
train_step=tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

#variable train_loss is used to save the loss value of each iteration
train_loss =[]

#error calculation
error = tf.reduce_mean(y-y_)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(150):

        #k-fold cross validation
        kf=KFold(n_splits=10)
        for train_index,test_index in kf.split(data):
            X_train,X_test=data[train_index],data[test_index]
            Y_train,Y_test=label[train_index],label[test_index]
            w_,b_,_,l=sess.run(fetches=[w,b,train_step,loss], feed_dict={x: X_train, y_: Y_train,m_train:X_train.shape[0]})
            if i%10==0:
                print("Epoch {0} : error:{1}".format(i,sess.run(error,feed_dict={x:X_test,y_:Y_test})))
        train_loss.append(l)
        #print("Epoch {0} : loss{1}".format(i, l))

#visual degree of fitting
Y_pred = X_train * w_ + b_
plt.plot(X_train,Y_train,'bo',label="real data")
plt.plot(X_train,Y_pred,'r',label="pred")
plt.legend()
plt.show()

#Drawing the image of a loss function
plt.plot(train_loss,label="train_loss")
plt.legend()
plt.show()
