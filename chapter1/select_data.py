import numpy as np
from sklearn.preprocessing import Imputer
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
total_rooms,total_bedrooms,population,household=3,4,5,6
def select_data(data, rato):
    """

    :param data: 未被划分的数据
    :param rato: 训练集数据的比例
    :return:
    """
    random_index = np.random.permutation(data.shape[0])
    tain_num = int(len(random_index) * rato)
    train_index = random_index[:tain_num]
    test_index = random_index[tain_num:]
    """
    shape[0]获取二位数组的行数
    shape[1]获取数组的列数
    """
    return data.iloc[train_index], data.iloc[test_index]
def data_set(housing_data):
    """
    我们发现在这份数据当中当地人的收入中位数对于该地区的房价有着密切的联系，而大部分的收入中位数分布在2-5万元，少量的数据分布在6万元之后，
    我们需要采用合理的方法把数据大致分布在2-5范围内，方便我们进行分层抽样
    通过除以1。5在向上取整，然后取位于5以内的数据
    :param housing_data: 原始数据集
    :return:

    """
    housing_data["income_cat"]=np.ceil(housing_data["median_income"]/1.5)
    housing_data["income_cat"].where(housing_data["income_cat"]<5,5.0,inplace=True)
    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=20)
    #训练集和测数据数据的划分
    for train_index,test_index in split.split(housing_data,housing_data["income_cat"]):
        train_data=housing_data.iloc[train_index]
        test_data=housing_data.iloc[test_index]
    temp=housing_data["income_cat"].value_counts()/len(housing_data)
    housing_data["income_cat"].hist()
    plt.show()
def data_viual_show(housing_data):
    temp_data=housing_data.copy()
    temp_data.plot(kind="scatter",x="longitude",y="latitude",alpha=0.5,
                   s=housing_data["population"]/100,label="population",
                   c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True,
                   sharex=True)# plpha为透明度的值，取0-1之间，0为透明，1为
                               # 当cmap（color map）不为空的时候，则c（color）不再指示颜色，而是深浅
                               # 从图像上可以看出人口的密集情况和地理位置、房价有着密切的关系
    #不透明、0.5为半透明
    plt.show()
def coefficient(housing_data):
    attributes=["median_house_value","median_income","total_rooms","housing_median_age"]

    # 通过我们对数据的发现显示收入的中位数与房屋的价值是密切相关的
    # scatter_matrix(housing_data[attributes],figsize=(12,8))
    # plt.show()

    # 详细显示median_housw_value he
    housing_data.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.2)
    plt.show()
def None_data_processing(housing_data):
    """
    用法：DataFrame.drop(labels=None,axis=0, index=None, columns=None, inplace=False)

   在这里默认：axis=0，指删除index（行），因此删除columns（列）时要指定axis=1；
   注意axis=0是对列进行计算，因此如果求平均值，那么求得应该是每一列得平均值，如果是drop，那么对列进行操作，变化得是行，同理对行也是一样的。

   inplace=False，默认该删除操作不改变原数据，而是返回一个执行删除操作后的新dataframe；

   inplace=True，则会直接在原数据上进行删除操作，删除后就回不来了。
    实现对数据的清洗(缺省值的处理)
    :param housing_data:
    :return:
    """
    housing=housing_data.drop("median_house_value",axis=1)
    housing_label=housing_data["median_house_value"].copy()
    housing_num=housing.drop("ocean_proximity",axis=1)
    imp=Imputer(strategy="median")
    imp.fit(housing_num)
    x=imp.transform(housing_num)
    # print(x)
    #  house_tr=pd.DataFrame(x,columns=housing_num.columns)
    # print(house_tr)
    # print(x.info())

def None_number_processing(housing_data):
    # @1对于样本数据中的非数值数据进行处理，首先是转换成整数值，然后利用OnehotEncode转换成独热码
    house_n_num=housing_data["ocean_proximity"].copy()
    encoder=LabelEncoder()
    encoder.fit(house_n_num)
    array1=encoder.transform(house_n_num)
    # print(array1)
    encoder=OneHotEncoder()
    onehot=encoder.fit_transform(array1.reshape(-1,1))
    # reshape(a,b)如果a,b均为非负整数，则前一个表示每行元素的个数，后一个表示每列的个数，如果为-1（行变成列）
    # print(onehot.toarray())# 转换成稀疏矩阵（方便我们进行观察）

    # @2直接利用和 LabelBinarizer实现从文本数据到整数数据再到独热码转换
class Attribute_Add(BaseEstimator,TransformerMixin):
    def __init__(self,bedrooms_per_room=None):
        self.bedrooms_per_room=bedrooms_per_room
    def fit(self,x,y=None):
        return self
    def transform(self, x, y=None, ):
          # x=inputdata,y=targetdata
        rooms_per_househols = x[:,total_rooms] / x[:,household]
        rooms_per_population = x[:,population] / x[:,total_rooms]
        if self.bedrooms_per_room:
          bedrooms_per_room=x[:,total_rooms]/x[:,total_bedrooms]
          return np.c_[x,rooms_per_househols,rooms_per_population,bedrooms_per_room]
        else:
            return np.c_[x,rooms_per_househols,rooms_per_population]

class get_data(BaseEstimator,TransformerMixin):
    def __init__(self,attributes):
        self.attributes=attributes;
    def fit(self,x,y=None):
        return self
    def transform(self,x,y=None):
        return x[self.attributes].values
class MyOneHotEncode(BaseEstimator,TransformerMixin):
    def __init__(self,data=None):
        self.data=data
    def fit(self,x,y=None):
        return self
    def transform(self,x,y=None):
        encoder = LabelEncoder()
        array1 = encoder.fit_transform(x)
        onehotencode=OneHotEncoder()
        tp=onehotencode.fit_transform(array1.reshape(-1,1))
        # print(tp.toarray())
        return tp.toarray()






