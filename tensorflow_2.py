# coding:utf-8
"""
@Auther      ：han
@Date        ：2019/11/19 14:49
@FileName    ：tensorflow_2.py
@Email       ：2579312470@qq.com
@Description ：
"""
import  numpy as np
import  tensorflow as tf
from numpy.random import RandomState

x=tf.placeholder(tf.float32,(None,2),name="x_input")
y_=tf.placeholder(tf.float32,(None,1),name="y_input")
w1=tf.Variable(tf.random_normal([2,1],stddev=1,dtype=tf.float32,seed=1))
y=tf.matmul(x,w1)
less=10
more=1
loss=tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*more,(y_-y)*less))
optimizer=tf.train.AdamOptimizer(0.001).minimize(loss)
batch_size=10
N=128
sdm=RandomState(1)
X=sdm.rand(N,2)
Y=[[x1+x2 +sdm.rand()/10.0-0.05] for (x1,x2) in X]
print(len(Y[1]))
step=1000
with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)

    for i in range(step):
        start=(i*batch_size)%N
        end=min(start+batch_size,N)
        sess.run(optimizer,feed_dict={x:X[start:end],y_:Y[start:end]})

    print(sess.run(w1))
