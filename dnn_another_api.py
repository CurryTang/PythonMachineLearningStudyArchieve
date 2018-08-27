import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected

##### parameters #####
learning_rate = 0.001
epoch = 50
batch_size = 128

n_inputs = 28 * 28
num_classes = 10

hidden1 = 100
hidden2 = 20


X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None, num_classes), name='y')


def dnn_model(features, labels, mode):
    layer1 = tf.layers.dense(inputs=features['x'], units=hidden1, activation=tf.nn.relu)
    layer2 = tf.layers.dense(inputs=layer1, units=hidden2, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=layer2, rate = 0.5, training= mode == tf.estimator.ModeKeys.TRAIN )
    logits = tf.layers.dense(inputs=dropout, units=10)

    pred_classes = tf.argmax(logits, axis = 1)
    pred_probas = tf.nn.softmax(logits)
    # PREDICT MODE
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, pred_classes)
    loss_op = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=tf.cast(
                                                           labels, dtype=tf.int32
                                                       ))
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, tf.train.get_global_step())
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, loss = loss_op, train_op = train_op)


    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs



mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype = np.int32)
eval_data = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
mnist_classifier = tf.estimator.Estimator(model_fn=dnn_model, model_dir="/tmp/dnn")
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'x': train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True
)
mnist_classifier.train(input_fn=train_input_fn, steps=200)
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False
)
e = mnist_classifier.evaluate(test_input_fn)
print("Accuracy: {}".format(e['accuracy']))
