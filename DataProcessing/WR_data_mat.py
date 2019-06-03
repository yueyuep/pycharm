#读取.mat数据格式的数据并使用tensorflow中的softmax函数进行分类
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import scipy.io as scio
from tensorflow.examples.tutorials.mnist import input_data

def getdata():
    dataset = scio.loadmat("./mnist-original.mat")
    data = dataset.get("data").T
    label = (dataset.get("label")).reshape(-1, 1)
    print(type(data))
    onehotencoder = OneHotEncoder()
    # 如果不加 toarray() 的话，输出的是稀疏的存储格式，即索引加值的形式，也可以通过参数指定 sparse = False 来达到同样的效果

    label_encode = (onehotencoder.fit_transform(label)).toarray()
    train_data, test_data, train_label, test_label = train_test_split(data, label_encode, test_size=0.2,
                                                                      random_state=100)
    # print(type(dataset))
    return train_data,test_data,train_label,test_label

def batch_generator(data,label,num):
    l=len(data[0])#获取我们数据的函数（样本数量）
    index=np.random.randint(0,l+1,size=num)
    return data[index],label[index]

if __name__ == '__main__':
    train_data,test_data,train_label,test_label=getdata()
    session = tf.InteractiveSession()  # 构建一个session
    x = tf.placeholder(tf.float32, shape=[None, 784])
    w = tf.Variable(tf.zeros(shape=[784, 10]))
    b = tf.Variable(tf.zeros(shape=[10]))
    y = tf.nn.softmax(tf.matmul(x, w) + b)
    _y = tf.placeholder(tf.float32, shape=[None, 10])
    mloss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(_y), reduction_indices=[1]))  # 行求和，这里是我们的损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(mloss)
    tf.global_variables_initializer().run()  # 初始化全局变量
    # 迭代1000次
    for i in range(5000):
        batch_x, batch_y = batch_generator(train_data, train_label, 100)
        train_step.run({x: batch_x, _y: batch_y})
    predict = tf.equal(tf.argmax(y, 1), tf.argmax(_y, 1))  # 相等返回true,否则返回false，按行返回最大值所对应的索引值
    accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))  # 布尔值转换成0、1
    print(accuracy.eval({x: test_data, _y: test_label}))


