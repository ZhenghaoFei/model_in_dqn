
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

a = np.asarray([[1,2,3,4,5],[1,2,3,4,6],[1,2,3,4,7]])
a = tf.Variable(a, dtype=tf.float32)
b = np.asarray([1,2,4])
print b.shape
b = tf.Variable(b, dtype=tf.float32)
b = tf.one_hot(tf.cast(b, tf.int32), depth=5)
d = a*b
d = tf.reduce_sum(d, axis=1)
# d = tf.gather_nd(a, b)
print a 
print b

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    c = sess.run(a)
    print c
    c = sess.run(d)
    print c
