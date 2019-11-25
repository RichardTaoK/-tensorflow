# coding:utf-8
"""
@Auther      ：han
@Date        ：2019/11/22 10:05
@FileName    ：tfrecord.py
@Email       ：2579312470@qq.com
@Description ：
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#生成整数的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#生成字符串的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnist=input_data.read_data_sets('D:\python\pytorch_mofan\pytorch\mnist\\raw',dtype=tf.uint8,one_hot=True)
images=mnist.train.images
labels=mnist.train.labels

pixels=images.shape[1]
num_examples=mnist.train.num_examples

#输出tfrecord的地址
filename="path"
#创建一个writer来写tfrecord
writer=tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    #将图像矩阵转为一个字符串
    image_raw=images[index].tostring()
    # 将一个样例转为exampleProtocol  Buffer ，并将所有的信息写入这个数据结构
    example=tf.train.Example(features=tf.train.Feature(feature={
        'pixels':_int64_feature(pixels),
        'label':_int64_feature(np.argmax(labels[index])),
        'image_raw':_bytes_feature(image_raw)
    }))
    #将一个example写入tfrecord
    writer.write(example.SerializeToString())
writer.close()




#读文件
reader=tf.TFRecordReader()
#创建一个队列来维护输入文件列表
filename_queue=tf.train.string_input_producer(["path/to/out.tfrecords"])
#从文件中读出一个样例，
_,serialized_example=reader.read(filename_queue)
#解析读入的一个样例，若需读出多个用parser_example
features=tf.parse_single_example(
    serialized_example,
    features={
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64)
    }
)

image=tf.decode_raw(features['image_raw'],tf.uint8)
label=tf.decode_raw(features['label'],tf.int32)
pixels=tf.decode_raw(features['pixels'],tf.int32)
sess=tf.Session()
sess.run([image,label,pixels])