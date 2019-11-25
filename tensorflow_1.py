# coding:utf-8
"""
@Auther      ：han
@Date        ：2019/11/18 20:48
@FileName    ：tensorflow_1.py
@Email       ：2579312470@qq.com
@Description ：拟合数据
"""
import tensorflow as tf
import numpy as np
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

#变量
weight=tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases=tf.Variable(tf.ones([1]))
y=weight*x_data+biases
#损失函数
loss=tf.reduce_mean(tf.square(y-y_data))
#定义优化器，主要是减小loss
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)
#初始化所有的变量
init=tf.initialize_all_variables()
#开启一个session会话
sess=tf.Session()
sess.run(init)
for step in range(240):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(weight),sess.run(biases))
