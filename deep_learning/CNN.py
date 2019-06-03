from sklearn.datasets import load_sample_image
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
if __name__ == '__main__':
    china=load_sample_image("china.jpg")
    flower=load_sample_image("flower.jpg")
    datas=np.array([china,flower],dtype=np.float32)
    batch_size,height,width,channels=datas.shape
    #定义我我们的卷积核7*7
    filters=np.zeros(shape=(7,7,channels,2))
    filters[:,3,:,0]=1 #垂直方向的卷积核
    filters[3,:,:,1]=1 #水平方向的卷积核
    x=tf.placeholder(tf.float32,shape=(None,height,width,channels))
    """
    placeholder:占位符，可以理解为形式参数(placeholder 和feed_dict的使用方式)
    
    
    
    在训练神经网络时需要每次提供一个批量的训练样本，如果每次迭代选取的数据要通过常量表示，
    那么TensorFlow 的计算图会非常大。因为每增加一个常量，TensorFlow 都会在计算图中增加一个结点。
    所以说拥有几百万次迭代的神经网络会拥有极其庞大的计算图，而占位符却可以解决这一点，它只会拥有占位符这一个结点
    
    shape：数据形状。默认是None，也就是一维值。
           也可以表示多维，比如要表示2行3列则应设为[2, 3]。
           形如[None, 3]表示列是3，行不定。
           
           import tensorflow as tf
 
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)
 
with tf.Session() as sess:
  output = sess.run(x, feed_dict = {x :'Hello World', y:123, z:45.67})
  print(output)
  output = sess.run(y, feed_dict = {x :'Hello World', y:123, z:45.67})
  print(output)
  output = sess.run(z, feed_dict = {x :'Hello World', y:123, z:45.67})
print(output)
    """
    # convolution=tf.nn.conv2d(x,filters,strides=[1,2,2,1],padding="SAME")
    # with tf.Session() as sess:
    #     output=sess.run(convolution,feed_dict={x:datas})
    # plt.imshow(output[1,:,:,1],cmap="gray")
    # plt.show()
    pooling=tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    with tf.Session() as sess:
        output=sess.run(pooling,feed_dict={x:datas})
        # plt.imshow(output[0].astype(np.uint8))
        plt.imshow(datas[0,:,:,:].astype(np.uint8))
        plt.show()

