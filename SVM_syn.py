from __future__ import print_function

import tensorflow as tf
import sys
import time
import numpy as np


# cluster specification
parameter_servers = ["localhost:2222"]

workers = ["localhost:2223", 
			"localhost:2224",
			"localhost:2225"]

cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})
#cluster = tf.train.ClusterSpec({"worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_float("delay", 0, "delay for simulation")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(
    cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index)

tm = time.localtime()

# config
batch_size = 100
learning_rate = 0.01
training_step = 1000
train_time = 6.0

logs_path = 'mnist/SVM/'
model_path = 'model/SVM/' + str(tm.tm_mon) + str(tm.tm_mday) + str(tm.tm_hour) + str(tm.tm_min) + '/'

# load mnist data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

train_images = mnist.train.images
train_labels = mnist.train.labels

test_images = mnist.test.images
test_labels = mnist.test.labels

# preprocessing for label, odd numbers with label 1, and even numbers with label -1
SVM_train_labels = np.reshape((2 * (train_labels % 2) - 1), (-1,1))
SVM_train_labels.dtype = 'int8'
SVM_test_labels = np.reshape((2 * (test_labels % 2) - 1), (-1,1))
SVM_test_labels.dtype = 'int8'


# number of test data points
num_test = mnist.test.num_examples


tf.reset_default_graph()

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
	
	# Between-graph replication
	with tf.device(tf.train.replica_device_setter(
		worker_device="/job:worker/task:%d" % FLAGS.task_index,
		cluster=cluster)):
		
		# count the number of updates
		global_step = tf.get_variable(
            name = 'global_step',
            shape = [],
            dtype = tf.int32,
            initializer = tf.constant_initializer(0),
			trainable = False)
			

		# input images
		with tf.name_scope('input'):
			# None -> batch size can be any size, 784 -> flattened mnist image
			x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
			# target 10 output classes
			y = tf.placeholder(tf.float32, shape=[None, 1], name="y-input")

		# model parameters will change during training so we use tf.Variable
		tf.set_random_seed(1)
		with tf.name_scope('parameters'):
			W = tf.Variable(tf.random_normal(shape=[784,1]))
			b =tf.Variable(tf.random_normal(shape=[1,1]))
			
		C = 1.0

		with tf.name_scope('output'):
			pred = tf.add(tf.matmul(x, W), b)
		

		# specify cost function
		with tf.name_scope('loss_function'):
			# this is our cost
			loss = C * tf.reduce_mean(tf.maximum(0.,1.0-tf.multiply(pred, y))) + 0.5 * tf.reduce_sum(tf.square(W))

		# specify optimizer
		with tf.name_scope('train'):
			# optimizer is an "operation" which we can execute in a session
			grad_op = tf.train.GradientDescentOptimizer(learning_rate, use_locking = True)
			
			rep_op = tf.train.SyncReplicasOptimizer(
		        grad_op,
				replicas_to_aggregate=len(workers),
				total_num_replicas=len(workers),
				use_locking=True)
			
			train_op = rep_op.minimize(loss, global_step=global_step)
 			
		with tf.name_scope('Accuracy'):
			# accuracy
			accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.sign(pred), y), tf.float32))
		
		#create the hook to handle initialization and queue
		sync_replicas_hook = [rep_op.make_session_run_hook((FLAGS.task_index == 0), num_tokens=0)]
			

		# create a summary for our cost and accuracy
		tf.summary.scalar("loss", loss)
		tf.summary.scalar("accuracy", accuracy)

		# merge all summaries into a single "operation" which we can execute in a session 
		summary_op = tf.summary.merge_all()
		init_op = tf.global_variables_initializer()
		
		
		print("Variables initialized ...")

		sess_config = tf.ConfigProto(
      		allow_soft_placement=True,
      		log_device_placement=False)

		
		with tf.train.MonitoredTrainingSession (
			master = server.target,
			is_chief = (FLAGS.task_index == 0),
			#checkpoint_dir = 'checkpoint/',
			hooks = sync_replicas_hook,
			config = sess_config,
			#max_wait_secs = 20
			) as mon_sess:

			if(FLAGS.task_index == 0):
				mon_sess.run(init_op)
			
			begin_time = time.time()
			frequency = 100
			
			# perform training cycles
			start_time = time.time()
	
			while not mon_sess.should_stop():
		
				# number of batches in one epoch
				batch_count = int(mnist.train.num_examples/batch_size)

				count = 0
				for i in range(batch_count):
				
					batch_x, batch_y0 = mnist.train.next_batch(batch_size)
					batch_y0 = 2 * (batch_y0 % 2) - 1
					batch_y0.dtype='int8'
					batch_y = np.reshape(batch_y0, (batch_size, 1))
		
					# perform the operations we defined earlier on batch
					_, cost, summary, step = mon_sess.run(
													[train_op, loss, summary_op, global_step], 
													feed_dict={x: batch_x, y: batch_y})
					count += 1
					
					time.sleep(FLAGS.delay)
					'''
					if count % frequency == 0 or i+1 == batch_count:
						elapsed_time = time.time() - start_time
						if count % frequency == 0:
							freq = frequency
						else:
							freq = frequency / 2
						
						print("node: %d" % (FLAGS.task_index),
									" Step: %d," % (step+1),  
									" Batch: %3d of %3d," % (i+1, batch_count), 
									" Cost: %.4f," % cost, 
									" AvgTime: %3.2fms" % float(elapsed_time*1000/freq))
						
						count = 0
						start_time = time.time()
					
					if step >= training_step: break
				
				if step >= training_step: break
				'''
					start_time = time.time()
					if step >= 300: break
				
				if step >= 300: break
				

			print("Test-Accuracy: %2.2f" % mon_sess.run(accuracy, feed_dict={x: test_images, y: SVM_test_labels}))
			print("Final Cost: %.4f" % mon_sess.run(loss, feed_dict={x: train_images, y: SVM_train_labels}))
			print("Total Time: %3.2fs" % float(time.time() - begin_time))
			
			print("node %d is done" % (FLAGS.task_index))
