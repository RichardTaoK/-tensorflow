### 深度学习的发展

在20年前，计算机就能够通过软件打败人类的象棋大师，但现如今还不能够像人一样，进行自然语言处理、图像识别等操作。

# 第三章 TensorFlow入门

### 创建一个session的过程

1. 创建一个session，所有的运算都默认跑在这个session中，不同session之间的数据与运算都是相互独立的
2. 创建一个Placeholder，即输入数据的地方
   - 第一个参数是数据类型
   - 第二个参数是shape=[None,784],None代表不限条数的输入，784代表每条输入是784维的

### 3.1 TensorFlow 计算模型——计算图

#### 3.1.1 计算图的概念

所有`TensorFlow`的程序都可以通过计算图的形式来表示，这是基本计算模型

#### 3.1.2 计算图的使用

1. 在`tf`中，系统会自动维护一个默认的计算图，通过`tf.get_default_graph`函数可以获取当前默认的计算图

2. 可以通过`tf.Graph`函数来生成新的计算图，不同计算图上的张量和运算不会共享

   ```
   import tensroflow as tf
   g1=tf.Graph()
   with g1.as_default():
   	v=tf.get_variable("v",initializer=tf.zeros_initializer(shape=[1]))
   #在计算图中读取v的取值
   with tf.Session(graph=g2) as sess：
   	tf.global_variables_initializer().run()
   	with tf.variable_scope("",reuse=True):
   		pirnt(sess.run(tf.get_variable("v"))
   ```

   ```
   
   ```

3. `tf.Graph.device`来指定运行计算的设备

   ```
   with tf.Graph.device('/gpu:0'):result=a+b
   ```

   ```
   
   ```

## 3.2 TensorFlow 数据模型——张量

### 3.2.1 张量的概念

张量：就是多维数组，其中零阶张量表示标量，也就是一个数；一阶张量是向量，即一维数组

张量并不直接参与运算，只是对tf中运算结果的引用。

- 一个张量主要保存三个属性：名字（计算节点的第几个输出）、维度shape、类型type
- `TensorFlow`会对参与运算的所有张量进行类型的检查，当发现类型不匹配是会报错
- `TensorFlow`支持14中不同的类型
- 可以通过`result.get_shape()`来获取张量的维度信息

1. 张量的使用情况：
   - 对中间计算结果的引用
   - 张量可以用来获得计算结果`tf.Session.run(result)`

### 3.3 `TensorFlow `运行模型——会话

1. 创建会话`sess=tf.Session`
2. `sess.run()`
3. `sess.close()`

或者

4. `with tf.Session() as sess:`

#### 3.4.3 神经网络参数与`TensorFlow`变量

定义变量要用到variable，变量Variable的作用就是宝尊和更新神经网络中的参数

**Tensorflow随机数生成**

- `tf.random_normal`   正太分布，平均值，标准差，取值类型
- `Tf.random_uniform`  均匀分布
- `tf.random_gamma`
- `tf.truncated_normal`  正太分布，如果随机出来的值偏离就会被重新随机分布

tf常数生成函数

- `tf.zeros`

- `tf.fill`  ` tf.fill([2,3],9)=[[9,9,9],[9,9,9]]`
- `tf.ones`
- `tf.constant`

`tf.global_variables_initializer()`将所有变量初始化

`tf.global_variables()`可以拿到所有的当前计算图上的所有变量



`tensorflow`提供placeholder机制用于提供输入数据，placeholder相当于定义了一个位置，这个位置中的数据在程序运行时再指定，只需要通过placeholder传入tf计算图中

期现象传播

```
x=tf.placeholder(tf.float32,shape=(1,2),name="ipnput")
w1=tf.random_normal([2,3],stddev=1)
w2=tf.random_normal([2.3],stddev=1)
a=tf.matmul(w1,x)
y=tf.matmul(a,w2)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

sess.run(y,feed_dict{x:[[0.7,0.9]]})
```

`feed_dict`是一个字典，在字典中需要给出每个用到的placeholder的值，如果某个需要的placeholder没哟被指定取值，那么程序在运行时会报错。

**训练神经网络的过程：**

- 定义神经网络的结构和前向传播的输出结果
- 定义损失函数以及选择反向传播优化的算法
- 生成会话并且在训练数据上反复运行反向传播优化算法，物理神经网络的结构如何变化，这三个步骤不变

# 第四章 深层神经网络

- 4.1 会介绍深层神经网络比浅层神经网络好在哪里
- 4.2 介绍如何设定损失函数也就是优化目标，会介绍几种常用的损失函数
- 4.3 讲解反向传播函数
- 4.4 优化中会遇到的几种问题

## 4.1 深度学习与深层神经网络

1. 激活函数
   - ReLu函数 f(x)=max(x,0)
   - sigmoid函数  
   - tanh函数

目前tensorflow提供了七种不同的非线性激活函数，`tf.nn.relu,tf.sigmoid,tf.tanh`

## 4.2 损失函数定义

神经网络模型的效果以及优化的目标是通过损失函数来定义的。我们不仅要知道有哪些损失函数还要知道如何根据具体问题来定义损失函数

**补充**

1. 因为一直不知道global_step的作用，因为learning_rate的作用，所以global_step代表每次的epoch数，所以在设置learning_rate的时候总是加上global_step

2. `tf.Variable`以及`tf.get_variable()`

   - `tf.Variable`初始化是直接传入initial_value的，我们一般这样初始化

     ```
     tf.Variable(initial_value=tf.random_normal(shape=[200,100],stddev=1.0),trainable=True)
     ```

   - `tf.get_variable(name="weight",shape=[200,100],dtype=tf.float32,initializer=tf.random_normal_initializer(stddev=0.1))`

   - `tf.Variable`出现了命名重名，系统会自动处理，但是`tf.get_variable()`是不会处理的，会直接报错的

   - 在共享变量时，会使用`tf.get_variable()` ,在创建变量时，会先查看有没有，有就报错；没有创建；若创建过，`reuse=False`会报错，解决方法是用`tf.reset_default_graph()`,若创建过且reuse=True,就会共用，综合来看，使用`reuse=tf.AUTO_REUSE`，那么没有创建就会重新创建，创建了就会共享该变量。

3. 为了方便管理变量，`tf`还有一个变量管理器，叫做`tf.variable_scope`

   ```
   import tensorflow as tf
   
   with tf.variable_scope("scope1"):
   	w1=tf.get_variable("w1",shape=[])
   	w2=tf.Varibale(0,name="w2")
   with tf.variable_scope("scope2"):
   	w1_p=tf.get_variable("w1",shape=[])
   	w2_p=tf.Varibale(0,name="w2")
   print(w1 is w1_p,w2 is w2_p)###True  False
   ```

   

### 1.  **损失函数**   分类问题通病

- 分类问题希望解决的是将不同的样本分到事先定义好的类别中。常常设置n个输出节点，其中n为类别的个数，数组中的每一个维度对应一个类别，在理想情况下，如果一个样本属于类别k，那么这个类别所对应的输出节点的输出值就是1，其他节点的输出值为0.
- **交叉熵**刻画了输出向量和期望向量之间概率分布的距离，![image-20191119111407171](C:\Users\han\AppData\Roaming\Typora\typora-user-images\image-20191119111407171.png)

- `softmax`将神经网络前向传播得到的结果变成了概率分布。原始神经网络的输出被用作置信度来生成新的输出，而新的输出满足概率分布的所有要求，导致一个样例为不同类别的概率分别是多大。这样就把神经网络的输出变成了一个概率分布，从而可以通过交叉熵来计算预测的概率分布和真实答案的概率分布之间的距离

通过`Tensorflow`实现交叉熵，代码如下：

```
cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
#y_是正确结果，y是预测结果
#tf.clip_by_value()函数是将一个张量中的数值限制在一个范围之内，此题是(1e-10,1)之间，这样可以避免一些运算错，将小于1e-10d的都换成1e-10,大于1的都换成1
##矩阵乘法是tf.matmul,但是数字相乘是*，是对应位置上的每个元素相乘
```

`Tensorflow`将交叉熵和`softmax`回归一起用

```
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y) 
```



通过`Tensorflow`来实现军方误差损失函数：

```
mse=tf.reduce_mean(tf.square(y_-y))
```

### 2. **梯度下降**使用batch_size的原因

梯度下降若有好几个最低点，那么可能到达不了最优点，除了不能获得最优点之外，梯度下降的计算时间太长。

- 要在全部训练数据上最小化损失，要是数据集太大，那么每一轮迭代中都需要计算全部训练数据上的损失函数。
- 为了极速训练过程，可以使用随机梯度下降的算法，不在全部训练数据集上计算损失函数，而是每一轮迭代中，随机优化某一条训练数据上的损失函数，但这样可能无法得到局部最优点
- 为了综合，会使用每次计算一小部分训练数据的损失函数，这一小部分数据被称为一个batch。这样可以大大减小收敛所需要的迭代次数，同时可以是收敛到的结果更加接近梯度下降的效果。

下面是所有的神经网络训练的过程

```
batch_size=n
x=tf.placeholder(tf.float32,shape=(batch_size,2),name="x_input")
y=tf.placeholder(tf.float32,shape=(batch_size,1),name="y_input")

loss=...
optimizer=....minimize(loss)
with tf.Session() as sess:
	init=tf.global_variables_initializer()
	sesss.run(init)
	for i in range(epoch):
		current_x,current_y=...
		sess.run(optimizer,feed_dict={x:current_x,y:current_y})
```

### 3. **学习率的设定**

`tensorflow`有一种更加灵活的学习率设置方法——支书衰减发`tf.train.exponential_decay`,可以随着迭代的继续逐步减小学习率，使得模型在训练后期更加稳定

**decayed_learning=learning_rate*$decay rate^{global step/decay steps}$ **

```
decayed_learning=learning_rate*decay_rate^(global_step/decay_steps)
#decay_rate为衰减系数
#decay_steps为衰减速度
```

下面这段代表使用学习率的衰减

```
global_step=tf.Variable(0)
learning_rate=tf.train.exponential_decay(0.1,global_step,100,0.96,staircase=True)
#参数分别为基础的学习率，global_step代表当前迭代的轮数，过完所有训练数据需要的迭代次数，学习率衰减速度
learning_step=tf.train.GradientDescentOptimizer(learning_rete).minimize(loss,global_step=global_Step)
#因为制定了staircase，所以没训练100轮学习率乘以0.96

```

### 4. **过拟合问题**

- 正则化：就是在损失函数中加入刻画模型复杂程度的指标，分为L1正则化和L2正则化

  假设损失函数式L，那么在优化是不是直接优化L函数，而是优化L+ $\alpha$ R(w),其中R(w)刻画的是模型的复杂程度，而入表示模型复杂损失在总损失中的比例。一般来说模型复杂度只有权重w决定。常用的正则化有

  1. L1正则化：`R(w)=||w||=sum|`$W_i$|

  2. L2正则化： R(w)=(||w||)^2= sum|$W_i^2$|

  无论是哪一种都是希望通过限制权重的大小，**使得模型不能任意拟合训练数据中的随机噪音**

  `loss=tf.reduce_mean(tf.square(y_-y))+tf.contrib.layers.l2_regularizer(lambda)(w)`

  这个是简单的带L2正则化的损失函数

### 5. 滑动平均模型

在采用随机梯度算法训练神经网络时，使用滑动平均模型在很多应用中都可以在一定程度上提高模型在测试数据上的表现

衰减率=min{decay,(1+num_update)/(10+num_update)}

`show_variable=decay*shadow_variable+(1-decay) *variable`decay一般设置为0.99或者0.999等非常接近1的数

`tf.train.ExponentialMovingAverage`

```
#定义一个变量用于计算滑动平均，这个变量的初始值是0
v1=tf.Variable(0,dtype=tf.float32)
#这里step变量模拟神经网络中迭代的轮数，可以用与动态控制虽简陋
step=tf.Variable(0,trainable=False)
#定义滑动平均类，初始化衰减率为0.99
ema=tf.train.ExponentialMovingAverage(0.99,step)
#定义一个操作，每次执行这个操作时这个列表中的变量会被更新
op=ema.apply([v1])
with tf.Session() as sess:
	init=tf.global_variables_initializer()
	sess.run(init)
	
	sess.run([v1,ema.average(v1)])  ##0,0
	sess.run(tf.assign(v1,5))   #降赋值给v1
	sess.run(op)#衰减率为  min{0.99,(1+step)/(10+step)=0.1}=0.1
	#所以v1的滑动平均为0.1*0+(1-0.1)*5=4.5,此时v1=5
	sess.run([v1,ema.average(v1)])   #5,4.5
	
	sess.run(tf.assign(step,10000))
	sess.run(tf.assign(v1,10))
	sess.run(op)   #decay=min{0.99,(1+step)/(10+step)=0.999}=0.99
	#0.99*4.5+0.01*10=4.555
	sess.run([v1,ems.average(v1)])   #10,4.555
```



# 第五章 `MNIST`数据集

## 5.1 `MNIST`数据集处理

1. 60000张训练集+10000张测试集，每一张图片代表了0-9中的一个数字。图片的大小都是28*28的

   ```
   from tensorflow.examples.tutorials.mnist import input_data
   mnist=input_data.read_data_sets("/path/to/mnist",one_hot=True)
   #会有train,valiation,test三个数据集
   
   
   同时
   mnist.train.next_batch(batch_size)
   ```

    

   

   

## 5.2 TensorFlow中的变量初始化函数

![image-20191119205234975](C:\Users\han\AppData\Roaming\Typora\typora-user-images\image-20191119205234975.png)





## 5.3 模型的持久化

1. 模型的保存

   ```
   saver=tf.train.Saver()
   with tf.Session() as sess:
   	sess.run(init)
   	saver.save(sess,'model.ckpt')
   	saver.restore(sess,"model.ckpt")#加载模型
   #以上这段代码会保存三个文件
   1.是model.ckpt.meta文件，它保存了TensorFlow计算图的结构
   2.是model.ckpt 保存了每一个变量的取值
   3.是checkpoint文件，保存了一个目录下所有的模型文件列表
   ```
   
2. tf提供了convert_variables_to_constants函数，可以将计算图中的变量及其取值通过常量的方式保存

   ```
   import 
   from tensorflow.python.framework import graph_util
   
   v1=tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
   v2=tf.Variable(tf.constant(2.0,shape=[1]),name="v2")
   result=v1+v2
   
   init=tf.global_variables_initializer()
   with tf.Session as sess:
   	sess.run(init)
   	graph=tf.get_default_graph().as_graph_def()
   	out_graph=graph_util.convert_variables_to_constants(sess,graph,['add'])
   	with tf.gfile.GFile("path.pb","wb") as f:
   		f.write(out_graph.SerializeToString())
   ```







# 第六章 卷积神经网络

神经网络分为卷积神经网络和全连接神经网络还有循环神经网络

## 6.1 卷积神经网络简介

1. 为什么全连接神经网络无法很好的处理图像数据

   使用全连接神经网络处理图像的最大问题在于全连接层的参数太多。**参数增多除了导致计算速度减慢，还很容易导致过拟合问题**，而卷积神经网络使用滤波，可以共享参数，大大减小了参数量

2. 卷积神经网络的5中结构
   - 输入层
   - 卷积层：卷积层视图将神经网络中的每一个小块进行更加深入地分析从而得到抽象程度更高的特征。而经过卷积层处理过的节点矩阵会变得很深。
   - 池化层：池化层不会改变矩阵的深度，但是可以缩小矩阵的大小。**池化操作可以认为是将一张分辨率较高的图片转化为分辨率较低的图片**，通过池化层，可以进一步缩小最后全连接层中节点的个数，从而减小网络中参数的个数
   - 全连接层：主要来完成最后的分类任务
   - softmax层：softmax层主要用于分类问题，通过softmax层，可以得到当前样例属于不同种类的概率分布情况。

## 6.2 卷积层

1. 卷积核可以称为内核kernel或者过滤器filter，过滤器可以将神经网络上的、一个子节点矩阵转化为下一层神经网络上的一个单位节点矩阵

   ![image-20191120112632085](C:\Users\han\AppData\Roaming\Typora\typora-user-images\image-20191120112632085.png)

2. 卷积矩阵的大小

   out=($in_{length}$-$filter_{length}$+1)/$stride_{length}$

3. tf.nn.conv2d提供了一个卷积的算法。

   - 第一个参数为batch

   - 第二个参数是卷积层的权重

   - 第三个参数提供的是一个长度为4的数组

   - 最后一个参数是填充padding其中，TensorFlow提供SAME和VALID两种选择，其中SAME表示添加全0填充，VALID表示不添加

     ```
     conv=tf.nn.conv2d(input,filter_weight,strides=[5,5,3,16],padding='SAME')
     
     bias=tf.nn.bias_add(conv,biases)
     
     activated_conv=tf.nn.relu(bias)
     ```

     

     

## 6.3 池化层

池化层可以非常好的减小矩阵的尺寸，从而减少最后全连接层的参数，使用池化层既可以加快计算速度还可以防止过拟合

分类：**最大池化**，**平均池化**

```
pool=tf.nn.max_pool(actived_conv,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
###第一个参数表示四维矩阵
#第二个参数表示过滤器的尺寸，虽然给出的是一个长度为4的一维数组，但是这个数组的第一个和最后一个数必须是1.这意味着池化层不能改变深度，最多是[1,2,2,1]  [1,3,3,1]
#第三个参数表示步长，而且第一维和最后一维只能是1
#第四个same表示全0填充，valid表示不使用全0 填充
```



## 6.4 LENet5

- 1. 第一层 卷积层：filter=[5,5,1,6],padding='Valid',strides=[1,1,1,1]
  2. 池化层：卷积核：[1,2,2,1]，步长=[1,2,2,1]
  3. 卷积层：filter=[5,5,1,16],'valid',1=[1,1,1,1]
  4. 池化层：卷积核：[1,2,2,1],[12,2,1]
  5. 全连接层
  6. 全连接层
  7. 全连接层

## 6.5 迁移学习

迁移学习就是将一个问题上训练好的模型通过简单的调整时期使用于一个新的问题。

1. 可以将inceptionv3中训练好的模型中所有卷积层的参数，只是替换最后一层全连接层。在最后一层之前的网络层称为瓶颈层。即特征提取层
2. 通过迁移学习，可以使用少量训练数据在短时间内训练出效果不错的神经网络模型





# 第七章 图像数据处理

## 7.1 Tfrecord输入数据格式

1. tfrecord数据都是通过tf.train.Example Protocol Buffer的格式存储的

2. 下面是将mnist数据集转为tfrecords

   ```
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
   ```

3. 当数据量较大时，可以将数据写入多个TFrecords，进行读文件

   ```
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
   ```

## 7.2 图像处理

- `tf.image.decode_jpeg(image_raw_data)`#对图像进行JPEG的格式解码从而得到图像对应的三维矩阵

- `tf.image.encode_jpeg(image_raw_data)`#将表示一张图像的三维矩阵重新按照jpeg格式编码并存入文件中，打开这张图像可以得到和原始图像一样的图像

- `tf.image.resize_images(image,[size,size],mothod=0)`图像大小调整#0代表双线性插值法、1代表最近邻居发、2代表双三次插值法、3代表面积插值法

- `tf.image.convert_iamge_dtype(image_data,dtype=tf.float32)`将0-255的像素整数值转为0-1范围内的实数值

- `tf.image.resize_image_with.crop_or_pad(image,1000,1000)`函数来调整图像进行裁剪或者填充,第一个参数表示图片，二三个参数表示其裁剪的大小

- `tf.image.flip_up_down(image)`#上下翻转

- `tf.image.flip_left_right`#水平翻转

- `tf.image.transpose_image(img)`#沿对角线翻转

- `tf.image.random_flip_up_down(img)`#以50%的概率上下翻转图像

- `tf.image.random_left_right(img)`

  ![image-20191122105559280](C:\Users\han\AppData\Roaming\Typora\typora-user-images\image-20191122105559280.png)

- 对比度

  ![image-20191122105627363](C:\Users\han\AppData\Roaming\Typora\typora-user-images\image-20191122105627363.png)

- 色相

  ![image-20191122105645697](C:\Users\han\AppData\Roaming\Typora\typora-user-images\image-20191122105645697.png)

  ![image-20191122105727505](C:\Users\han\AppData\Roaming\Typora\typora-user-images\image-20191122105727505.png)

- 饱和度

  ![image-20191122105755193](C:\Users\han\AppData\Roaming\Typora\typora-user-images\image-20191122105755193.png)

- 图像标准化

  ![image-20191122105832173](C:\Users\han\AppData\Roaming\Typora\typora-user-images\image-20191122105832173.png)

- 随机截取图像

  ![image-20191122110321826](C:\Users\han\AppData\Roaming\Typora\typora-user-images\image-20191122110321826.png)

## 7.3 多线程输入数据处理框架

1. 流程：

   ![image-20191122110458111](C:\Users\han\AppData\Roaming\Typora\typora-user-images\image-20191122110458111.png)

   2. 利用数据集来读取数据有是哪个基本步骤
      - `tf.data.Dataset.from_tensor_slices()`从数据集是从一个张量中构建的
      - `tf.one_shot_iterator`遍历数据集
      - `tf.get_next()`从便离去在读取数据张量，作为其他计算图的输入部分



# 第十一章 Tensorboard可视化

## 11.1 

1. 运行tensorboard

   ```
   tensorboard --logdir=/path/to/log
   然后打开浏览器输入localhost:6006
   ```

# 第十二章 TensorFlow计算加速

## 12.1 使用GPU

1. 可以通过`tf.device`来指定运行每一个操作的设备，这个设备可以是本地cpu可以是GPU，还可以是远程服务器

- 所有的cpu均称为`tf.device(/cpu:0)`作为名称

- 一台机器上不同GPU的名称是不同的，第n个GPU称为`/gpu:n`

- 可以使用log_device_placement出纳室来打印，每一个运算的设备。

  ```
  a=tf.constant([1,2,3],shape=[3],name="a")
  b=tf.constant([1,2,3],shape=[3],name="b")
  result=a+b
  sess.run(config=tf.ConfigProto(log_device_placement=True))
  #参数的意思是会将每一个操作的设备输出到屏幕。最后不仅可以看到最后的计算结果还可以看到CPU：0这样的输出
  print(sess.run(result))
  ```

- tfsorflow会优先将运算设置在GPU上的。有时候，尽管机器有四个GPU，但是tf仍然会优先放到/gpu:0上的

  ```
  import tensorflow as tf
  with tf.device('/cpu:1'):
  	a=tf.constant([1,2,3],shape=[3],name="a")
  	b=tf.constant([1,2,3],shape=[3],name='b')
  whith tf.device('/gpu:2'):
  	c=a+b
  sess.run(config...)
  sess.run(c)
  ```

  ![image-20191122201403161](C:\Users\han\AppData\Roaming\Typora\typora-user-images\image-20191122201403161.png)

- 但是并不是所有的变量都是可以放到GPU上来执行的，如在GPU上，tf.Variable操作值支持实数型(float16,float32,double)的参数，若给定的参数是整数型的，是不支持在GPU上运行的。为了避免这个问题，我们在会话中设置config的时候，可以设置`allow_soft_placement`参数设置为True，这样子，当无法在gpu执行的时候tf会自动将其转到CPU上的

- tf是默认占用所有的GPU以及每个GPU的所有显存的，但是我们可以通过设置**`CUDA_VISIBLE_DEVICES`**,环境变量来控制，只使用部分GPU

  ```
  tf支持在程序中设置环境变量，以下代码展示了如何在程序中设置这些环境变量。
  
  os.venvision["CUDA_VISIBLE_DEVICES"]="2"
  ```

#### 按需分配gpu

虽然tf会一一次性占用一个GPU的所有显存，但是TensorFlow也支持分配GPU的显存。是的一块GPU上可以同时运行多个任务。下面给出了TensorFlow动态分配显存的方法

```
config=tf.ConfigProto()
#让tf按需分配显存
config.gpu_options.allow_growth=True
#或者直接按固定的比例分配，一下代码会占用所有可使用GPU的40%显存
config.gpu_options.per_precess_gpu_memory_fraction=0.4
session=tf.Session(config=config,...)
```

## 12.3 多GPU并行

给出具体的tf代码，在一台机器上的多个GPU上并行训练深度学习模型

# Tensorflow项目流程

![20190331112657924](D:\笔记\tensorflow学习笔记\20190331112657924.png)

# Tensorflow的模块介绍

## 1. tf.contrib.layers

在`tf.contrib.layers`内部，有许多产生layer操作及其相关权重和偏差变量的函数。这些大部分都是用来构建不同深度学习架构的。也有些函数是提供归一化，卷积层，dropout层（注：Dropout是在训练过程中以一定概率1-p将隐含层节点的输出值清0），‘one-hot’编码等。下面来粗略浏览一下：

- **`tf.contrib.layers.optimizers`模块**：`tf.contrib.layers.optimizers`包括的优化器有Adagrad，SGD，Momentum等。它们用来解决数值分析的优化问题，比如，优化参数空间寻找最优模型-- - 
- **`tf.contrib.layers.regularizers`模块**：`tf.contrib.layers.regularizers`包括的正则化有L1规则化和L2规则化。规则化经常被用来抑制模型训练时特征数过大导致的过拟合（overfitting）问题；有时也作为Lasso回归和Ridge回归的构建模块；
- **`tf.contrib.layers.initializers`模块**：`tf.contrib.layers.initializers`一般用来做模型初始化。包括深度学习在内的许多算法都要求计算梯度来优化模型。随机初始化模型参数有助于在参数空间中找到最优参数解。TensorFlow提供的有Xavier初始化器，用来在所有层中保持梯度大体相同；
- **`tf.contrib.layers.feature_column`模块**：`tf.contrib.layers.feature_column`提供函数（比如，bucketing/binning，crossing/compostion，和embedding）来转换连续特征和离散特征；
- `tf.contrib.layers.embedding`模块：`tf.contrib.layers.embedding`转化高维分类特征成低维、密集实数值向量。

## 2. tf.cast()

在tensorflow中常用到`tf.cast()`,通俗来说，就是框架中的类型转换函数，`tf.cast(pred,"float")`,将pred预测值转换成浮点类型，因为tf中值支持float类型

# 补充：多线程管理

1. `tf.train.Coordinator`

   tf是支持多线程的，可以子啊同一个会话中创建多个线程并执行。在Session中的所有线程 都必须能被同步终止，异常必须能被正确捕获并报告，会话终止的时候， 队列必须能被正确地关闭 

   `tf`提供了两个类来实现Session中多线程的管理：`tf.Coordinator`和`tf.QueueRunner`这两个类往往一起使用

   - `Coordinator`类用来管理在Session中的多个线程，可以用来同时停止多个工作线程并且向那个在等待所有工作线程终止的程序报告异常，该线程捕获到这个异常之后就会终止所有线程。使用 `tf.train.Coordinator()`来创建一个线程管理器（协调器）对象。

   - `QueueRunner`类用来启动tensor的入队线程，可以用来启动多个工作线程同时将多个tensor（训练数据）推送入文件名称队列中，具体执行函数是 `tf.train.start_queue_runners` ， 只有调用 `tf.train.start_queue_runners `之后，才会真正把tensor推入内存序列中，供计算单元调用，否则会由于内存序列为空，数据流图会处于一直等待状态。

     ![20180401184032574](D:\笔记\tensorflow学习笔记\20180401184032574.gif)
   
   1. 调用 `tf.train.slice_input_producer`，从 本地文件里抽取tensor，准备放入Filename Queue（文件名队列）中;
   
      ```
      slice_input_producer(tensor_list, num_epochs=None, shuffle=True, seed=None,
                               capacity=32, shared_name=None, name=None)
      #第一个参数 tensor_list：包含一系列tensor的列表，表中tensor的第一维度的值必须相等，即个数必须相等，有多少个图像，就应该有多少个对应的标签。
      第二个参数num_epochs: 可选参数，是一个整数值，代表迭代的次数，如果设置 num_epochs=None,生成器可以无限次遍历tensor列表，如果设置为 num_epochs=N，生成器只能遍历tensor列表N次。
      第三个参数shuffle： bool类型，设置是否打乱样本的顺序。一般情况下，如果shuffle=True，生成的样本顺序就被打乱了，在批处理的时候不需要再次打乱样本，使用 tf.train.batch函数就可以了;如果shuffle=False,就需要在批处理时候使用 tf.train.shuffle_batch函数打乱样本。
      第四个参数seed: 可选的整数，是生成随机数的种子，在第三个参数设置为shuffle=True的情况下才有用。
      第五个参数capacity：设置tensor列表的容量。
      第六个参数shared_name：可选参数，如果设置一个‘shared_name’，则在不同的上下文环境（Session）中可以通过这个名字共享生成的tensor。
      第七个参数name：可选，设置操作的名称。                   
      ```
   
      
   
   2. 如果shuffe是true时调用` tf.train.batch`，如果shuffle是False时，调用`tf.train.shuffle_batch`打乱函数，从文件名队列中提取tensor，使用单个或多个线程，准备放入文件队列;
   
      ```
      batch(tensors, batch_size, num_threads=1, capacity=32,          enqueue_many=False, shapes=None, dynamic_pad=False,          allow_smaller_final_batch=False, shared_name=None, name=None)
      #----------------------------------------------------------#
      ###第一个参数tensors：tensor序列或tensor字典，可以是含有单个样本的序列[image,label];
      第二个参数batch_size: 生成的batch的大小;
      第三个参数num_threads：执行tensor入队操作的线程数量，可以设置使用多个线程同时并行执行，提高运行效率，但也不是数量越多越好;
      第四个参数capacity： 定义生成的tensor序列的最大容量;
      第五个参数enqueue_many： 定义第一个传入参数tensors是多个tensor组成的序列，还是单个tensor;
      第六个参数shapes： 可选参数，默认是推测出的传入的tensor的形状;
      第七个参数dynamic_pad： 定义是否允许输入的tensors具有不同的形状，设置为True，会把输入的具有不同形状的tensor归一化到相同的形状;
      第八个参数allow_smaller_final_batch： 设置为True，表示在tensor队列中剩下的tensor数量不够一个batch_size的情况下，允许最后一个batch的数量少于batch_size， 设置为False，则不管什么情况下，生成的batch都拥有batch_size个样本;
      第九个参数shared_name： 可选参数，设置生成的tensor序列在不同的Session中的共享名称;
      第十个参数name： 操作的名称;
      ```
   
      
   
   3. 调用` tf.train.Coordinator()` 来创建一个线程协调器，用来管理之后在Session中启动的所有线程;
   
   4. 调用`tf.train.start_queue_runners`, 启动入队线程，由多个或单个线程，按照设定规则，把文件读入Filename Queue中。函数返回线程ID的列表，一般情况下，系统有多少个核，就会启动多少个入队线程（入队具体使用多少个线程在`tf.train.batch`中定义）;这个是配套`tf.train.slice_input_producer`来的#`tf.train.slice_input_producer`定义了样本放入文件名队列的方式，包括迭代次数，是否乱序等，要真正将文件放入文件名队列，还需要调用`tf.train.start_queue_runners` 函数来启动执行文件名队列填充的线程，之后计算单元才可以把数据读出来，否则文件名队列为空的，计算单元就会处于一直等待状态，导致系统阻塞。
   
   5. 文件从 Filename Queue中读入内存队列的操作不用手动执行，由tf自动完成;
   
   6. 调用`sess.run `来启动数据出列和执行计算;
   
   7. 使用 `coord.should_stop()`来查询是否应该终止所有线程，当文件队列（queue）中的所有文件都已经读取出列的时候，会抛出一个` OutofRangeError` 的异常，这时候就应该停止Session中的所有线程了;
   
   8. 使用`coord.request_stop()`来发出终止所有线程的命令，使用`coord.join(threads)`把线程加入主线程，等待threads结束。
   
   ## 猫狗大战的运用
   
   ```
   # coding:utf-8
   """
   @Auther      ：han
   @Date        ：2019/11/18 20:48
   @FileName    ：tensorflow_1.py
   @Email       ：2579312470@qq.com
   @Description ：对数据进行处理，生成batch
   """
   
   import tensorflow as tf
   import numpy as np
   import os
   
   def get_file(image_path,shuffle=True):
       image_train=[]
       image_label=[]
       for item in os.listdir(image_path):
           image_path=image_path+'/'+item
           label=item.split('.')[0]
           if os.path.isfile(image_path):
               image_train.append(image_path)
           else:
               raise ValueError('文件夹中有非文件项')
           if label=='cat':
               image_label.append(0)
           else:
               image_label.append(1)
       image_train=np.asarray(image_train)
       image_label=np.asarray(image_label)
       if shuffle:
           idx=np.arange(len(image_train))
           np.random.shuffle(idx)
           image_train=image_train[idx]
           image_label=image_label[idx]
       return image_train,image_label
   def get_batch(train_list,image_size,batch_size,capacity,is_random=True):
       intput_queue = tf.train.slice_input_producer(train_list, shuffle=False)
       image_train=tf.read_file(intput_queue[0])
       image_train=tf.image.decode_jpeg(image_train,channels=3)#格式是jpg
       image_train=tf.image.resize_images(image_train,[image_size,image_size])
       #归一化为0-1
       image_train=tf.cast(image_train,tf.float32)/255.
       image_label=intput_queue[0]
       if is_random:
           image_train_batch,label_train_batch=tf.train.shuffle_batch([image_train,image_label],batch_size,capacity=capacity,min_after_dequeue=100,num_threads=2)
       else:
           image_train_batch,label_train_batch=tf.train.batch([image_train,image_label],batch_size=batch_size,num_threads=2,capacity=capacity)
       return image_train_batch,label_train_batch
   if __name__=="__main__":
       image_path="data/train"
       train_list=get_file(image_path,True)
       image_train_batch,label_train_batch=get_batch(train_list,256,32,200,False)
       sess=tf.Session()
       coord=tf.train.Coordinator()
       threads=tf.train.start_queue_runners(sess=sess,coord=coord)
       try:
           for step in range(10):
               if coord.should_stop():#查询是否应该终止所有线程
                   break
               image_batch,label_batch=sess.run([image_train_batch,label_train_batch])
               if label_batch[0] == 0:
                   label = 'Cat'
               else:
                   label = 'Dog'
               plt.imshow(image_batch[0]), plt.title(label)
               plt.show()
       except tf.errors.OutOfRangeError:
           print('Done.')
       finally:
           coord.request_stop()
   
       coord.join(threads=threads)#把线程加入主线程，等待threads结束。
       sess.close()
   ```
   



## 随机打乱数据集,划分数据集

1.

```
if shuffle:
        idx=np.arange(len(image_train))
        np.random.shuffle(idx)
        image_train=image_train[idx]
        image_label=image_label[idx]
```

2. d 

   ```
   #划分数据集
   val_percentage=10
   test_percentage=10
   change=np.random.randint(10)
   if chance<val_percentage:
   	val_images.append(image_value)
   	val_images.append(current_label)
   elif (chance<test_percentage+val_percentage):
   	test_images.append(image_value)
   	test_images.append(current_label)
   else:
   	train_images.append(image_value)
   	train_images.append(current_label)
   #打乱来获得更好的训练效果
   state=np.random.get_state()
   np.random.shuffle(train_images)
   np.random.set_state(state)
   np.random.shuffle(train_labels)
   ```

   