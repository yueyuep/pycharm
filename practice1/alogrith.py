import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
import matplotlib

eta=0.05
n_iterations=1000
m=100
n_epochs=50
theta=np.random.randn(2,1)
t0,t1=5,50

x=np.random.rand(100,1)
y=4+3*x+np.random.randn(100,1)
x_c=np.c_[x,np.ones((100,1))]

def group_SGDRregressor():
    """
    线性回归问题，利用批量梯度下降算法（还有一种利用最小二乘法来求解a,b得值），梯度下降算法，比较适合于特征值比较多的情况，求得a,b得值
    eta：学习率，学习率决定每次得步长，theta=theta-eta*gradients
    gradients:梯度
    theta:求得参数
    """
    theta = np.random.randn(2, 1)
    sgd_regressor = SGDRegressor(random_state=47, loss="squared_loss")
    sgd_regressor.fit(x, y)

    line_regressor = LinearRegression()
    line_regressor.fit(x, y)

    sgd_predicts = sgd_regressor.predict(x)
    line_predicts = line_regressor.predict(x)

    for iteration in range(n_iterations):
        gradients = 2 / m * x_c.T.dot(x_c.dot(theta) - y)
        theta = theta - eta * gradients

    plt.scatter(x, y, marker="*", alpha=0.8)
    plt.plot(x, theta[0, 0] * x + theta[1, 0], 'r--')
    plt.plot(x, sgd_predicts, 'g--')
    plt.plot(x, line_predicts + 0.1, 'b-')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    print(theta)
def schedule(t):
    return t0/(t+t1)
def Schedule_sgd():
    theta = np.random.randn(2, 1)
    for epochs in range(n_epochs):
        """
        控制学习率得变换，一共迭代50次
        """
        for i in range(n_iterations):
            random_index = np.random.randint(100)
            xi = x_c[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]
            """"
            每次只是采用了一个样本数据
            """
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = schedule(m*epochs+i)
            theta=theta-eta*gradients
    plt.scatter(x, y, marker="*", alpha=0.8)
    plt.plot(x, theta[0, 0] * x + theta[1, 0], 'r--')
    plt.show()
    print("a:\nb:\n",theta[0, 0],theta[1, 0])
def letter_start():
    sgd_regressor=SGDRegressor(n_iter=1,penalty=None,eta0=0.0005,learning_rate="constant",warm_start=True)
    """
    learning_rate="constant",学习率需要设定一个常数
    warm_start=True,则训练模型的时候，能够继续 从上次训练的地方开始
    """
    mininum_val_error=float("inf")
    best_iteration=None
    best_model=None
    for iterations in range(n_iterations):
        sgd_regressor.fit(x,y)
        y_predicts=sgd_regressor.predict(x)
        error=mean_squared_error(y,y_predicts)
        if error<mininum_val_error:
            mininum_val_error=error
            best_iteration=iterations
            best_model=clone(sgd_regressor)
    print("minnum_val_error:\nbest_iteration:\nbest_model:\n",mininum_val_error,best_iteration,best_model)







if __name__ == '__main__':
    #Schedule_sgd()
    #letter_start()
    group_SGDRregressor()


