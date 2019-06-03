import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as Lda
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
def get_data(url):
    dataframe = pd.read_excel(url, sheetname="Data")
    # 注意pandas中的切片处理常用
    """
    dataframe["label"]直接按照标签来取数据
    dataframe.loc[1:2,['a','b']]也是按照标签来进行取数据
    dataframe.iloc[1:,2:6] 完全按照索引来取数据，前闭后开
    df.loc[‘image1’:‘image10’, ‘age’:‘score’]  前后都闭
    """

    #数据的预处理部分
    tmp=dataframe.dropna(axis=0)   #去掉我们的原始数据中的空值数据
    attribute = tmp.iloc[0:, 7:]#对列进行操作
    attribute=attribute.drop([97]) #去掉标签不明的数据
    label = tmp["Class"]
    label=label.drop([97])



    #标准化数据
    st=StandardScaler()
    attribute=st.fit_transform(attribute)
    #返回属性、标签
    return attribute,label

def LDA(x,y):
    #类别标签
    label_class=['granite','diorite','marble','limestone','breccia','slate']
    #训练数据和测试数据的划分
    train_data,test_data,train_label,test_label=train_test_split(x,y,test_size=0.2,random_state=100)

    #计算每一个类别的均值向量
    mean_vec=[]
    for i in label_class:
        mean_vec.append(np.mean(train_data[train_label==i],axis=0))
    #计算类内散度矩阵sw
    f=18
    sw=np.zeros((f,f))
    for i,vec in zip(label_class,mean_vec):
        for row in train_data[train_label==i]:
            tmp,tvec=row.reshape(-1,1),vec.reshape(-1,1)
            sw=sw+np.dot((tmp-tvec),(tmp-tvec).T)
    #计算类间矩阵
    mean_all=np.mean(train_data,axis=0)
    sb=np.zeros((f,f))
    for i,vec in zip(label_class,mean_vec):
        n=train_data[train_label==i].shape[0]
        mean_all=mean_all.reshape(-1,1)
        vec=vec.reshape(-1,1)
        sb = sb + n * np.dot((mean_all - vec), (mean_all - vec).T)
    values,eigvec=np.linalg.eig(np.linalg.inv(sw).dot(sb))

    eigpairs=[(np.abs(values[i]),eigvec[:,i]) for i in range(len(values))]
    eigpairs=sorted(eigpairs,key=lambda x:x[0],reverse=True)#按照第一个关键字进行排序
    print("-------------------从大到小排序的特征向量-------------------")
    for eig_vec in eigpairs:
        print(eig_vec[0])#打印逆序排列的特征值

    #m为前两个最大特诊值对应的特征向量
    m=np.hstack((eigpairs[0][1].reshape(-1,1).real,eigpairs[1][1].reshape(-1,1).real))
    #print(eigpairs[0][1][:, np.newaxis].real)


    #transform_data为我们降维后的数据
    transform_data=train_data.dot(m)


    #绘制我们线性判别分析后的数据分布情况。
    colors=['g','r','b','c','m']
    marks=['+','+','+','+','+']
    for i,c,m in zip(np.unique(train_label),colors,marks):
        #l=len(transform_data[train_label==i])
        #y=np.zeros((l,1))
        plt.scatter(transform_data[train_label==i][:,0],transform_data[train_label==i][:,1],c=c,marker=m)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    #讲我们的标签换成数字特征。(考虑为何不可)
    # train_label[train_label=="granite"]=0
    # train_label[train_label == "diorite"] = 1
    # train_label[train_label == "marble"] = 2
    # train_label[train_label == "limestone"] = 3
    # train_label[train_label == "breccia"] = 4
    # train_label[train_label == "slate"] = 5
    lb=LabelEncoder();
    train_label=lb.fit_transform(train_label.values)
    test_label=lb.fit_transform(test_label.values)
    train_label.reshape(-1,1)
    test_label.reshape(-1,1)








    #调用我们的sklearn中的线性判别分析进行特诊选取
    lda=Lda(n_components=2)
    print("-------------------标签数组------------------")
    data_train = lda.fit_transform(train_data, train_label)
    print(train_label)

    print("-----------------将维后的数据---------------------")
    print(data_train)


    data_test=lda.fit_transform(test_data,test_label)
    r = LogisticRegression()
    r.fit(data_train,train_label)


    #r为评估器，输出我们的预测标签
    r.predict(data_test)
    print("------------预测的准确率------------")
    print(r.score(data_test,test_label))






if __name__ == '__main__':
    url = "./datas/ROCKS.XLS"
    x,y=get_data(url)
    LDA(x,y)






