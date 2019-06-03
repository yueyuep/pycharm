import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import StratifiedShuffleSplit
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")
def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

if __name__ == '__main__':
    datas=fetch_mldata("mnist-original",data_home="./")
    dataset=datas["data"]
    label=datas["target"]
    one_encoder=OneHotEncoder()
    label_one_hot=(one_encoder.fit_transform(label.reshape(-1,1))).toarray()
    #矩阵如何按列进行合并
    shuffe_split=StratifiedShuffleSplit(n_splits=1,rain_size=0.8,test_size=0.2)
    for train_index,test_index in shuffe_split.split(dataset,label):
        #划分训练集
        train_data=dataset[train_index]
        train_label=label[train_index]
        #划分测试集
        test_data=dataset[test_index]
        test_label=label[test_index]
    x=tf.placeholder(dtype=tf.float32,shape=(None,784))
    y=tf.placeholder(dtype=tf.float32,shape=(None,10))
    x_iamge=tf.reshape(x,shape=(-1,28,28,1))


    b_conv1=bias_variable([32])#第一层的卷积、池化操作
    w_conv1=weight_variable([5,5,1,32])
    h_conv1=tf.nn.relu(conv2d(x_iamge,w_conv1)+b_conv1)
    h_pool1=max_pool(h_conv1)

    #第二层的卷积、池化操作
    w_conv2=weight_variable([5,5,1,64])
    b_conv2=bias_variable([64])
    h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
    h_pool2=tf.nn,max_pool(h_conv2)
 


    








