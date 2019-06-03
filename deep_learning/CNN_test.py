import tensorflow as tf
import numpy as np
dataset=tf.Variable(tf.ones(shape=(1,28,28,1),dtype=tf.float32))
batch_size,height,width,channels=dataset.shape
filters=tf.Variable(tf.ones(shape=(5,5,1,1)))
x=tf.placeholder(tf.float32,shape=(1,height,width,1))
convolution=tf.nn.conv2d(x,filters,strides=[1,5,5,1],padding="VALID")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ouput=sess.run(convolution,feed_dict={x:dataset})