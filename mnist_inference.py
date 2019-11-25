# coding:utf-8
"""
@Auther      ：han
@Date        ：2019/11/20 9:55
@FileName    ：mnist_inference.py.py
@Email       ：2579312470@qq.com
@Description ：定义了前向传播的过程以及神经网络中的参数
"""

import tensorflow as tf

input=784
output=10
layer_1=500
def get_weight_variable(shape,regularizer):
    weights=tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer!=None:##加入正则化
        tf.add_to_collection('losses',regularizer(weights))
    return weights
def inference(input_tensor,regularizer):
    #第一层
    weights=get_weight_variable([input,layer_1],regularizer)
    biases=tf.get_variable("biases",shape=[layer_1],initializer=tf.constant_initializer(0.0))
    layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    #第二层
    with tf.variable_scope('layer2'):
        weights=get_weight_variable([layer_1,output],regularizer)
        biases = tf.get_variable("biases", shape=[output], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
    return layer2

