import matplotlib.pyplot as plt
import math
from sympy.plotting import plot
import numpy as np
from sympy import *
r=symbols('r')
def kernal_function(x,h):
    #定义核函数为正态分布函数
    return 1/(sqrt(2*pi))*exp((-(((r-x)/h)*((r-x)/h))/2))
    #return math.exp(r-x)+thea

if __name__ == '__main__':
    y1=0
    y2=0
    np.random.seed(15)
    T= np.random.randn(150)
    #T=np.array([-2.1,-1.3,-0.4,1.9,5.1,6.5,])
    l=len(T)
    q1=np.percentile(T,25) #四分之一点
    q3=np.percentile(T,75)
    iqr=q3-q1
    variances=np.var(T)

    #根据平均积分平均误差最小得到h
    h=1.06*variances*Pow(l,-1/5)

    #根据分位点确定的h值
    A=min(variances,iqr/1.34)
    h_opt=0.9*A*Pow(l,-1/5)

    # 根据平均积分平均误差最小得到h
    h = 1.06 * variances * Pow(l, -1 / 5)
    for t in T:
        y2 += kernal_function(t, h_opt)
        y1 += kernal_function(t, h)
    p1=plot(y2/((l-1)*h_opt))
    #p2=plot(y1/(l*h))
    #plt.hist(T,bins=35)
    #plt.show()








