import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def gen_clusters():
    mean1 = [0, 5, 2]
    cov1 = [[25, -1,7], [-1, 4,-4],[7,-4,10]]
    data = np.random.multivariate_normal(mean1, cov1, 800)
    return np.round(data, 4)


def save_data(data, filename):
    with open(filename, 'w') as file:
        for i in range(data.shape[0]):
            file.write(str(data[i, 0]) + ',' + str(data[i, 1]) +','+str(data[i,2])+ '\n')


def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            data.append([float(i) for i in line.split(',')])
    return np.array(data)


def show_scatter(data):

    #原始的数据表示
    x,y,z=data.T[0],data.T[1],data.T[2]
    #pca主成分分析
    mean_data=np.mean(data,axis=0)#按照列计算均值
    #print(mean_data)
    cen_data=data-mean_data
    #np.cov 每一行代表一个变量，每一列代表这些变量的观测值
    cov_data=np.cov(cen_data.T)
    value,vector=np.linalg.eig(cov_data)
    #选取0、1
    #value1=value[:,[0,1]]
    ject_feature1=np.dot(cen_data,vector[:,[0,1]])
    #print(ject_feature1)

    # 选取0、2
    #value1 = value[:, [0,2]]
    ject_feature2= np.dot(cen_data,vector[:, [0,2]])

    # 选取1、2
    #value1 = value[:, [1,2]]
    ject_feature3 = np.dot(cen_data,vector[:, [1,2]])

    mfigure=plt.figure()
    ax1 = mfigure.add_subplot(221,projection='3d')
    ax2 = mfigure.add_subplot(222)
    ax3 = mfigure.add_subplot(223)
    ax4 = mfigure.add_subplot(224)
    ax1.scatter(x,y,z,c='r',s=2)
    ax2.scatter(ject_feature1[:,0],ject_feature1[:,1],s=2)
    ax3.scatter(ject_feature2[:,0],ject_feature2[:,1],s=2)
    ax4.scatter(ject_feature3[:,0],ject_feature3[:,1],s=2)
    plt.show()

    #调用我们的sklearn.decomposion.PCA实现过程
    # mpca=PCA(n_components=3)
    # new_data=mpca.fit_transform(data)
    # print(new_data)
    # print(mpca.explained_variance_ratio_)
    # ax1.scatter(x,y,z,c='r',s=2)
    # ax2.scatter(new_data[:,0],new_data[:,1],s=2)
    # ax3.scatter(new_data[:,0],new_data[:,2],s=2)
    # ax4.scatter(new_data[:,1],new_data[:,2],s=2)
    plt.show()

    # plt.scatter(x, y)
    # plt.axis()
    # plt.title("scatter")
    # plt.xlabel("x")
    # plt.ylabel("y")


data = gen_clusters()
#print(data)
save_data(data, '3clusters.txt')
d = load_data('3clusters.txt')
show_scatter(d)
