
from functools import reduce
import pandas as pd
import numpy as np
"""
input_num: 输入向量的维数，比如这里实现的是两个数的与运算，因此每组数据应该包括两个值，同时每个值对应一个权重。
activator:激活函数，下面的f函数即为定义的激活函数。
"""
class perceptron(object):
    def __init__(self, input_num, activator):#这里的
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0
    def __str__(self):
        #可以理解为java中的ToString方法，这里是重写了改方法
        return('weight\t:%s\nbias\t:%f\n'%(self.weights,self.bias))
    def predict(self,input_vec):
        return self.activator(
            reduce(lambda a,b:a+b,list(map(lambda x,w:float(x)*float(w),self.weights,input_vec)),0.0)+self.bias
            # f(x1*w1+x2*w2+b)指代我们的预测函数。
        )
    """
    self 可以理解为类所属的变量，也可以认为是类的实例，只有类的实例
    才能够调用类的成员方法
    类：定义对象的模板
    对象(类的实例对象)
    ps：除了类的静态变量或者是静态方法，其他的方法都需要通过类的实例
    对像进行调用
    
    class Test:
    def ppr(self):
        print(self)
        print(self.__class__)

     t = Test()
     t.ppr()
     执行结果：
     <__main__.Test object at 0x000000000284E080>
     <class '__main__.Test'>

     """


    def train(self,input_vecs,labels,iteration,rate):
        for i in range(iteration):
            self.__one_iteration(input_vecs,labels,rate)
    def __one_iteration(self,input_vecs,labels,rate):
        samples=zip(input_vecs,labels)
        for (input_vecs,labels) in samples:
            output=self.predict(input_vecs)
            self.__update__weights(input_vecs,output,labels,rate)
    def __update__weights(self,input_vec,output,labels,rate):
        delta=float(labels)-output
        self.weights=list(map(lambda x,w:w+rate*delta*float(x),input_vec,self.weights))
        self.bias+=rate*delta
def f(x):
    if x>0:
        return 1
    else:
        return 0
def get_training_dataset():

    # input_vecs=[[1,1],[0,0],[1,0],[0,1]]
    # labels=[1,0,0,0]
    data = pd.read_csv("data1.csv", header=None)
    attribute = data.ix[1:, 0:5]
    labels = data.ix[1:, 6]
    attribute = np.array(attribute,dtype=float)
    labels = np.array(labels,dtype=float)
    return attribute,labels
def train_and_perceptron():
    p=perceptron(6,f)
    input_vecs,labels=get_training_dataset()
    p.train(input_vecs,labels,10000,0.001)
    return p
if __name__=='__main__':
   test_data,test_labels=get_training_dataset()
   and_perceptrom=train_and_perceptron()#位于一个模块下的代码必须缩进一样，否则会报错，严格的格式要求
   print(and_perceptrom)
   predict=[]
   for i in test_data:
       temp=and_perceptrom.predict(i)
       predict.append(temp)
   tmp=(test_labels-predict).tolist()
   print("准确率：%f"%(tmp.count(0)/len(tmp)))

   #print("预测标签；%d",and_perceptrom.predict(test_data))
   #rint()
   # print("1 and 0  the result is :%d\n" %and_perceptrom.predict([63,10.77,22.7,6.26,1,1]))
   # print("1 and 1  the result is :%d\n" %and_perceptrom.predict([80,12,21.77604,51.76,1,0]))
   # # print("0 and 0  the result is :%d\n" %and_perceptrom.predict([0,0]))
   # # print("0 and 1 the result is :%d\n" % and_perceptrom.predict([0, 1]))






















