import pandas as pd
import numpy as np
import math
from sklearn import datasets
from sklearn.base import BaseEstimator,TransformerMixin
def sigmod(x):
    return 1/(1+math.e**(-x))

def my_logic(data,label):
    """
    自定义的逻辑回归函数（利用梯度下降）
    :param x:
    :param y:
    :return:
    """
    data_one=np.c_[data,np.ones((150,1))]
    m=len(data)
    eta=np.random.rand(2,1)
    learning_rate=0.005
    n_iteration=1000
    for iteration in range(n_iteration):
        gradients=1/m*data_one.T.dot(sigmod(data_one.dot(eta))-label)
        eta=eta-learning_rate*gradients
    #print("a:\nb:\n",eta[0,0],eta[1,0])
    return eta




if __name__ == '__main__':
    """
    导入sklearn中的数据集，总共有150个样本数据，然后有4个特征值，分别是萼片的长度、宽度，
    花瓣的长度、宽度
    """
    data=datasets.load_iris()
    x=data["data"][:,3].ravel().reshape(-1,1)
    y=(data["target"]==2).astype(np.int).reshape(-1,1)
    eta=my_logic(x,y)
    test=np.array([[1.7],[1]])
    predict_score=1/(1+math.e**(-eta.T.dot(test)))




