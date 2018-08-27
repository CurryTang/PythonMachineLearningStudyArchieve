# Tensorflow distributed CNN

本文基于 [Original Repo](https://github.com/maxr1876/DistTF)

## Distributed Machine Learning 101
面临分布式机器学习的问题是，一个很自然的想法是将问题分解为多个layer,分给每个机器去处理每一个layer.但是这种方法对机器学习显然不适用，像神经网络这样的结构，只有通过之前的计算结果才能得到后面的计算结构。于是，我们在这里采用了一种单一parameter server,多个worker的结构:
![](https://tensorport-api-development.s3.amazonaws.com/filer_public_thumbnails/filer_public/0c/f7/0cf7471b-7cc3-414b-824b-919fddba7d63/data-parallelism.jpg__1109x900_q85_crop_subsampling-2_upscale.jpg)
图片援引https://clusterone.com/blog/2017/09/13/distributed-tensorflow-clusterone/

我们只保留一个model在parameter上，每次各个worker获得一个mini-batch的数据来进行训练，并将得到的gradient返回到parameter server.
接下来又会引出async与sync两种模型，目前对于这一方面的研究主要集中在如何同步数据上。有的paper提出了过几轮同步一次的方法来解决stale gradient的问题(由于训练速度不同，有时候用的weight还是好几轮之前的)，同时又能加快速度，有的paper甚至给出了一个可调的参数来控制这个更新的频率。
"When Edge Meets Learning: Adaptive Control for Resource-Constrained Distributed Machine Learning"这篇2018年发表的paper对这一领域有比较相近的论述，是很好的入门材料。

## async model



## 基本配置

使用tf.app.flags.DEFINE_string来进行命令行参数的定义
api文档中对这个函数没有详细的介绍，但根据源码可以推测出DEFINE_string有三个参数，第一项为命令行参数项的名称，第二项为默认值，第三项为注释
如果我们有如下的定义
``` python
tf.app.flags.DEFINE_integer("hidden_units", 100,
                            "Number of units in the hidden layer of the NN")
```
那我们可以在命令行参数中使用`--hidden_units 200 `来进行hidden layer的自定义。

## CNN 架构

接下来对原生的CNN架构进行进一步封装.

``` python
def weight_variable(shape):
  '''
	tf.truncated_normal returns random values from a truncated normal distribution. 
	The generated values follow a normal distribution with specified mean and standard 
	deviation, except that values whose magnitude is more than 2 standard deviations
	from the mean are dropped and re-picked.
	'''
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


```
* `truncated_normal`在normal的基础上有微小的变动，如果得到的值离mean太远，就将他丢掉重新取，所以最后得到的点会都集中在正态分布的中心区域附近

*  mean在默认情况下取0

``` python

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
```
* `tf.nn.conv2d(input,filter,strides,padding,use_cudnn_on_gpu=True,data_format='NHWC',dilations=[1, 1, 1, 1],name=None)` 

### 关于SAME
我们知道用以计算output_width与output_height有以下的公式

$$ \text{output width} = \dfrac{W-F_w+2P}{S_w}+1 $$

$$ \text{output height} = \dfrac{H-F_h+2P}{S_h}+1 $$ 


那么得到的值可能并不是整数,SAME的做法是做padding, 而valid是做ceiling就结束. 详情参阅[SAME and Valid](http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html)

### 关于stride
一开始完全搞不明白为啥这是四维的？后来发现这是tensorflow的一个奇怪设定，第一个和第四个参数分别为batch_increment与channel_increment(也就是一次跳几个btach与一次跳几个channel), 查阅资料后发现这两项必须设定为1. ksize也是同理.



## 实体代码

通过刚才的配置，我们可以初始化ClusterSpec配置来搭建一个server.
API文档如此描述一个server
> A tf.train.Server instance encapsulates a set of devices and a tf.Session target that can participate in distributed training. A server belongs to a cluster (specified by a tf.train.ClusterSpec), and corresponds to a particular task in a named job. The server can communicate with any other server in the same cluster.


那么整体的架构就是有一个总的cluster, 下属数个server(分别代表worker, parameter server的职能)
``` python
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  
  '''
  After creating a list of param servers and workers, create an instance 
  of tf.train.ClusterSpec. This defines what the cluster looks like, so 
  each machines are aware of both itelf and all the other machines.
  '''
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  '''
  In order to communicate with the other machines, a server must be created.
  It is important to provide the server with the ClusterSpec object as well
  as the name of the current job along with the task_index of the job. This
  way, the server is aware of each machine's role within the system.
  '''
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

```
接下来如果一个server是parameter server的职能，那我们什么都不用做，用server.join()使其堵塞即可;如果是一个worker, 那么我们要编写相应的逻辑代码
``` python
  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
```

接下来的问题是如何给每个server分配好相应的工作
### 关于replica_device_setter
根据Google的文档，我们是可以完全手动地分配这些device的
``` python

with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(...)
  biases_1 = tf.Variable(...)

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(...)
  biases_2 = tf.Variable(...)

with tf.device("/job:worker/task:7"):
  input, labels = ...
  layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
  logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
  # ...
  train_op = ...

with tf.Session("grpc://worker7.example.com:2222") as sess:
  for _ in range(10000):
    sess.run(train_op)
```
但是这样非常麻烦而且完全不必要，因为系统有自行分配的功能.

```
 with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/replica:0/task:%d" % FLAGS.task_index,
        cluster=cluster)):
```
这里简单谈一下in-graph replication 与 between-graph replication. 前者是只存一份模型在parameter server里，而后者是每个worker都在本地存一份模型。推荐使用后者，因为在replica多的时候后者效率更高。在这里我们选用replica_device_setter就是选用了后者。
```
 	  # global_step coordinates between all workers the current step of training
      global_step = tf.Variable(0, trainable=False)
      #More to come on is_chief...
      is_chief = FLAGS.task_index == 0
      
      '''
      A placeholder is a symbolic variable. It can be used as input to a 
      TensorFlow computation. The size 784 is due to the fact that all 
      input images are 28x28 pixels. Here None means that a dimension 
      can be of any length.
      '''
      x = tf.placeholder(tf.float32, [None, 784])
      y_ = tf.placeholder(tf.float32, [None, 10])
      
      #Read the input data
      mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

```
以上为一些基本的配置,None表示这里的length可以是任何值
### CNN model
最后我们需要配置的就是CNN模型
```
      ################################################################################################
      '''
      Here we initialize the weights and bias. Create a tensorFlow Variable, which
      represents a modifiable Tensor that will be passed from node to node in the 
      computational graph. W has a shape of [784, 10] because we want to multiply
      the 784-dimensional image vectors by it to produce 10-dimensional vectors 
      of evidence for the difference classes. b has a shape of [10] so it can be 
      added to the output.
      '''
      W = tf.Variable(tf.zeros([784, 10]))
      b = tf.Variable(tf.zeros([10]))
      
      '''
      At this point, we can begin to define the model. We define y to be the softmax 
      function. softmax accepts a computation that will be performed at each node of
      the graph.. In our case, this is multiplying x*W, and adding b to the result 
      of that multiplication.
      '''
      y = tf.nn.softmax(tf.matmul(x, W) + b
      ##################################################################################################
      ## 以上部分应当与CNN模型无关
      
      '''
      Initialize the first convolutional layer. The convolution will compute 32 features for each 5x5 
      image patch. Hence the first two parameters being 5. The 1 represents the number of channels in the image.
      As these are grayscale images, there is only one channel. Were these to be color images, there would be 3
      channels (rgb). The fourth argument is the number of channels to output. We will also need a bias vector
      with a component for each output channel.
      '''
      W_conv1 = weight_variable([5, 5, 1, 32])
      b_conv1 = bias_variable([32])
      
      '''
      To apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions corresponding 
      to image width and height, and the final dimension corresponding to the number of color channels.
      '''
      x_image = tf.reshape(x, [-1, 28, 28, 1])
      
      '''
      We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool. 
      The max_pool_2x2 method will reduce the image size to 14x14.
      '''
      h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
      h_pool1 = max_pool_2x2(h_conv1)
      
      '''
      Now initialize the second convolutional layer. This layer will be more dense, computing 64 features instead of 32.
      '''
      W_conv2 = weight_variable([5, 5, 32, 64])     
      b_conv2 = bias_variable([64])    
      h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
      h_pool2 = max_pool_2x2(h_conv2)
      
      '''
      Now it's time for the densely connected layer. This layer will have 1024 neurons to allow processing on
      an entire image. The images will now be 7x7 (caused by h_pool2).
      '''
      W_fc1 = weight_variable([7 * 7 * 64, 1024])
      b_fc1 = bias_variable([1024])
      h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
```
1. 有关tf.reshape 参见这个[stackoverflow的问答](https://stackoverflow.com/questions/41778632/why-is-the-x-variable-tensor-reshaped-with-1-in-the-mnist-tutorial-for-tensorfl)
2. 
