import pandas as pd
import numpy as np
import scipy.io as sio
import os
from matplotlib import pyplot as plt
import matplotlib
from sklearn.linear_model import SGDClassifier
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier

"""
利用fetch_mldata 读取，这个是在网上下载数据，（该网站访问不了，这里使用的本地数据读取）
"""
# minist_data=fetch_mldata('mnist-original',data_home='./')
# print(minist_data)
"""
利用scipy.io.loadmat()读取，不知道为啥读取的数据跟原来的数据是互为转置的
"""
mat_path = os.path.join('mldata', 'mnist-original.mat')
minist_data = sio.loadmat(mat_path)

x, y = minist_data['data'].T, minist_data['label'].T
y=y.ravel()
# some_digit=x[6000]
# # some_image=some_digit.reshape(28,28)
# # plt.imshow(some_image,cmap=matplotlib.cm.binary,interpolation="nearest")
# # plt.axis("off")
# # plt.show()
train_data = x[:60000]
train_label = y[:60000]
test_data = x[60000:]
test_label = y[60000:]
seed = np.random.permutation(60000)
Rtrain_data = train_data[seed]
Rtrain_label = train_label[seed]
Rtarin_label_1 = (Rtrain_label == 5)
sgd_classifier = SGDClassifier(random_state=47)
rand_tree_classifier=RandomForestClassifier(random_state=47)

def classficater():
    """
    一个简单的二分类器（没有分析数据正例和反例的比例）
    :return:
    """
    # sgd_classifier.fit(Rtrain_data,Rtarin_label_1)
    # print(Rtrain_data)
    # print(sgd_classifier.predict([some_digit]))# 注意数据的格式
    skfolds = StratifiedKFold(n_splits=3, random_state=42)
    for train_data_index, test_data_index in skfolds.split(
            Rtrain_data, Rtarin_label_1):
        clone_model = clone(sgd_classifier)
        TrainData = train_data[train_data_index]
        TrainLabel = (Rtarin_label_1[train_data_index])
        TestData = train_data[test_data_index]
        TestLabel = (Rtarin_label_1[test_data_index])
        clone_model.fit(TrainData, TrainLabel)
        predicts = clone_model.predict(TestData)
        n_true = sum(predicts == TestLabel)
        accuracy = n_true / (len(predicts))
        print(accuracy)
def mrog_curve():
    y_probas_forest=cross_val_predict(rand_tree_classifier,Rtrain_data,Rtarin_label_1,cv=3,method="predict_proba")
    """
    cross_val_predict(method="ptoba_predict")最后得出是一个矩阵，每一行代表一个样本数据，每一列代表一个类
    以本例子为例，因此每个位置代表改样本预测成每个类别的概率，二分类问题，一共有两列，第一列为反例，第二列为正例
    下局代码，我们取正例作为预测分数（后续补充）
    """
    y_probas_score=y_probas_forest[:,1]
    predicts=cross_val_predict(sgd_classifier,Rtrain_data,Rtarin_label_1,cv=3,method="decision_function")
    """
    FPR:假真例，反例被错误是被成正例
    TPR：真正例,正例被是被为正例子(召回率)
    rog：受试者感应曲线
    """
    t_FPR,t_TPR,t_thresholds=roc_curve(Rtarin_label_1,y_probas_score)
    FPR,TPR,thresholds=roc_curve(Rtarin_label_1,predicts)
    """
    1、随机梯度分类器所得到的ROC曲线
    2、随机深林分类器得到的ROC曲线
    """

    """
    绘制ROC曲线
    """

    plt.plot(FPR,TPR,linewidth=2,label="SGD_Classifier")
    plt.plot(t_FPR,t_TPR,linewidth=2,label="Random_forest")
    plt.plot([0,1],[0,1],'b--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="upper_left")
    plt.show()



def evaluate_classficater():
    """
    混淆矩阵，每一行代表实际的一个类别，每一列代表一个预测的类
          flase  true
   flase [[53487(TN)  1092(FP)]
   true  [ 1281(FN)  4140(TP)]]

   有53487非5被预测成非5，有1092个非5被预测成5
   有1281个5被预测成非5，有4140干扰5被预测成5

   准确率：
   accuracy=TP/(TP+FP)
   TPR(true positive rate)=TP/(TP+FN)
    :return:
    """
    predicts = cross_val_predict(
        sgd_classifier, Rtrain_data, Rtarin_label_1, cv=3)
    con_matrix = confusion_matrix(Rtarin_label_1, predicts)
    print(con_matrix)
    print("the accuracy:\t", precision_score(Rtarin_label_1, predicts))
    print("the recall rate:\t", recall_score(Rtarin_label_1, predicts))
    print("the f1_score(调和平均值)：\t", f1_score(Rtarin_label_1, predicts))
    # method的默认值为predict在这里即为最终的预测标签，但是这一次指定返回一个决策分数，而不是预测值。
    y_scores = cross_val_predict(sgd_classifier, Rtrain_data, Rtarin_label_1, cv=3, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(
        Rtarin_label_1, y_scores)
    mprecision_recall_curve(precisions, recalls, thresholds)


def mprecision_recall_curve(precisions, recalls, thresholds):
    plt.plot(thresholds, recalls[:-1], "b--", label="recalls")
    plt.plot(thresholds, precisions[:-1], "g-", label="precisions")
    plt.xlabel("thresholds")
    plt.legend(loc="upper_right")
    plt.ylim([0, 1])
    plt.show()
def multify_classficater():
    """
    通过二分类器来达到多分类的目的
    :return:
    """
    one_vs_one_class=OneVsOneClassifier(sgd_classifier)
    one_vs_one_class.fit(Rtrain_data,Rtrain_label)
    predict_label=one_vs_one_class.predict([Rtrain_data[40000]])
    print("实际的标签：\t预测的标签：\t分类器的数目：\t",Rtrain_label[40000],predict_label)


def multify_classficater_confus_matrix():
    predict_labels=cross_val_predict(sgd_classifier,Rtrain_data,Rtrain_label,cv=3)
    con_matrix=confusion_matrix(Rtrain_label,predict_labels)
    row_sum=con_matrix.sum(axis=1)
    con_matrix=con_matrix/row_sum
    np.fill_diagonal(con_matrix,0)
    plt.matshow(con_matrix,cmap=plt.cm.gray)
    plt.show()

def multi_labels_classficater():
    label1=(Rtrain_label>7)
    label2=(Rtrain_label%2==0)
    multi_labels=np.c_[label1,label2]
    k_neighbor=KNeighborsClassifier()
    k_neighbor.fit(Rtrain_data,multi_labels)
    predict=k_neighbor.predict([Rtrain_data[40000]])
    print("多标签预测结果：",predict)

if __name__ == '__main__':
     #evaluate_classficater()
     #mrog_curve()
     #ultify_classficater()
     #multify_classficater_confus_matrix()
     multi_labels_classficater()

