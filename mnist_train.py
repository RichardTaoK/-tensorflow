# coding:utf-8
"""
@Auther      ：han
@Date        ：2019/11/20 10:07
@FileName    ：mnist_train.py
@Email       ：2579312470@qq.com
@Description ：训练
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference


##配置神经网络的参数
batch_size=64
learning_rate_base=0.8
learning_rate_decay=0.99
regulation_rate=0.0001
epoch=30000
moving_average_rate=0.99

model_save_path="model"
model_name="model.ckpt"


def train(mnist):
    x=tf.placeholder(tf.float32,[None,mnist_inference.input],name="x_input")
    y_=tf.placeholder(tf.float32,[None,mnist_inference.output],name="y_input")
    regularizer = tf.contrib.layers.l2_regularizer(regulation_rate)
    #直接用前向传播
    y=mnist_inference.inference(x,regularizer)
    global_step=tf.Variable(0,trainable=False)
    #运动平均
    variable_averages=tf.train.ExponentialMovingAverage(moving_average_rate,global_step)
    op=variable_averages.apply(tf.trainable_variables())

    #交叉熵
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)

    #损失函数
    loss=cross_entropy_mean+tf.add_n(tf.get_collection("losses"))

    #学习率衰减
    learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,mnist.train.num_examples/batch_size,learning_rate_decay)

    #优化器操作
    optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    with tf.control_dependencies([optimizer,op]):
        train_op=tf.no_op(name='train')

    #初始化tf持久化类
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        import os
        for i in range(epoch):
            xs,ys=mnist.train.next_batch(batch_size)
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%1000==0:
                print("step:%d,loss:%g"%(step,loss_value))
            saver.save(sess, 'model/model.ckpt',global_step=global_step)
def main(argv=None):
    mnist=input_data.read_data_sets("D:\python\pytorch_mofan\pytorch\mnist//raw",one_hot=True)
    train(mnist)
if __name__ == '__main__':
    tf.app.run()