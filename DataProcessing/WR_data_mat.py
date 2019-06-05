#读取.mat数据格式的数据并使用tensorflow中的softmax函数进行分类
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import scipy.io as scio
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

#############################################################
                        #batch封装类,注意阅读
class DataSet(object):

    def __init__(self, images, labels, num_examples):

        self._images = images      #我们的训练数据

        self._labels = labels       #我们的训练标签

        self._epochs_completed = 0  # 完成遍历轮数

        self._index_in_epochs = 0  # 调用next_batch()函数后记住上一次位置

        self._num_examples = num_examples  # 训练样本数

    def next_batch(self, batch_size, fake_data=False, shuffle=True):

        start = self._index_in_epochs   #初始位置为0

        if self._epochs_completed == 0 and start == 0 and shuffle:  #第一次batch
            #获取我们训练数据的长度
            index0 = np.arange(self._num_examples)

            #print(index0)
            #打乱我们的数据
            np.random.shuffle(index0)

            #print(index0)
            #返回我们的数据
            self._images = np.array(self._images)[index0]

            self._labels = np.array(self._labels)[index0]
            print("-------第一次的batch----------")
            print(self._images)

            print(self._labels)




        if start + batch_size > self._num_examples: #这个是最后一次的batch，或者最后一次batch数据不够

            self._epochs_completed += 1

            rest_num_examples = self._num_examples - start

            images_rest_part = self._images[start:self._num_examples]

            labels_rest_part = self._labels[start:self._num_examples]

            if shuffle:
                index = np.arange(self._num_examples)

                np.random.shuffle(index)

                self._images = self._images[index]

                self._labels = self._labels[index]

            start = 0

            self._index_in_epochs = batch_size - rest_num_examples

            end = self._index_in_epochs

            images_new_part = self._images[start:end]

            labels_new_part = self._labels[start:end]

            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(

                (labels_rest_part, labels_new_part), axis=0)



        else:
            print("第%d次batch训练常数"%(self._epochs_completed))
            print("start数据是：%d"%(start))

            self._index_in_epochs += batch_size

            end = self._index_in_epochs
            print("end数据是：%d" % (end))
            return self._images[start:end], self._labels[start:end]


#############################################################


def getdata():
    dataset = scio.loadmat("./mnist-original.mat")
    data = dataset.get("data").T
    label = (dataset.get("label")).reshape(-1, 1)
    print(type(data))

    #########################################
    onehotencoder = OneHotEncoder()
    # 如果不加 toarray() 的话，输出的是稀疏的存储格式，即索引加值的形式，也可以通过参数指定 sparse = False 来达到同样的效果
    label_encode = (onehotencoder.fit_transform(label)).toarray()
    #train_data, test_data, train_label, test_label = train_test_split(data, label_encode, test_size=0.1,                                                                 random_state=100,shuffle=False)
    # print(type(dataset))



    train_data,train_label=data[0:55000,:],label_encode[0:55000,:]
    test_data,test_label=data[60000:70000,:],label_encode[60000:70000,:]


    return train_data,test_data,train_label,test_label

def batch_generator(data,label,num):
    l=len(data[0])#获取我们数据的函数（样本数量）
    index=np.random.randint(0,l,size=num)
    np.random.shuffle(index)
    return data[index],label[index]




if __name__ == '__main__':

    #################################################
    #                     使用自己处理好的数据
    #train_data,test_data,train_label,test_label=getdata()



    #################################################
    #                     使用内置处理好的数据

    #
    minist_data = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    test_data=minist_data.test.images
    test_label=minist_data.test.labels
    train_data=minist_data.train.images
    train_label=minist_data.train.labels




    ################################################
    session = tf.InteractiveSession()  # 构建一个session
    x = tf.placeholder(tf.float32, shape=[None, 784])
    w = tf.Variable(tf.zeros(shape=[784, 10]))
    b = tf.Variable(tf.zeros(shape=[10]))
    y = tf.nn.softmax(tf.matmul(x, w) + b)
    _y = tf.placeholder(tf.float32, shape=[None, 10])

    #行求和，这里是我们的损失函数
    mloss = tf.reduce_mean(-tf.reduce_sum(_y * tf.log(y), reduction_indices=[1]))




    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(mloss)
    tf.global_variables_initializer().run()  # 初始化全局变量
    # 迭代1000次

    for i in range(1000):
        #batch_x, batch_y=minist_data.train.next_batch(100)
        batch_x,batch_y=DataSet(train_data,train_label,55000).next_batch(100)
        #batch_x, batch_y = batch_generator(train_data, train_label, 1000)
        train_step.run({x: batch_x, _y: batch_y})
    predict = tf.equal(tf.argmax(y, 1), tf.argmax(_y, 1))  # 相等返回true,否则返回false，按行返回最大值所对应的索引值,按照行返回
    accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))  # 布尔值转换成0、1
    print(accuracy.eval({x: test_data, _y: test_label}))


