import os
from matplotlib import pyplot as plt
import pandas as pd
import tarfile
from select_data import *
import urllib
import numpy as np
from urllib.request import urlretrieve
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib


def download(url, savepath):
    """
    :param url: 下载路径
    :param savepath:保存路径
    :return:
    """
    filename = os.path.basename(url)
    filepath = os.path.join(savefile, filename)
    if not os.path.isdir(savefile):
        # 如果不存在我们设置的保存文件夹，则自动生成一个
        os.makedirs(savefile)
        # 下载文件
        urlretrieve(url, filepath)
        # 打开我们的文件
        downfile = tarfile.open(filepath)
        # 解压我们的文件
        downfile.extractall(savefile)
        # 关闭读写操作
        downfile.close()
        print("download finished")
    else:
        print("file has existed")


if __name__ == '__main__':
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
    savefile = "./chapter1/data"
    # load data
    download(url, savefile)
    # read data
    housing_data = pd.read_csv("chapter1/data/housing.csv", delimiter=',')
    ocean_data = housing_data["ocean_proximity"].value_counts()
    # ocean_data.head() #显示我们的前几行数据
    # ocean_data["ocean_proximity"].value_counts()#我们注意到ocean_proximity在我们的数据表中显示的同一个值，且为对象
    # 因此可以查看该对象属性值范围，以及每个属性值在数据表中出现的个数
    # test=housing_data.describe() #描述数值属性的概括
    # print(test)
    # hist 直方图，表示某一值出现的频数其中bins表示直方图条数的多少，bins越多，则条会越小(越窄)
    # ，直方图会容易的看出中位数
    # housing_data.hist(bins=50,figsize=(20,15))
    # figsize参数：指定绘图对象的宽度和高度，单位为英寸；dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80。
    # 因此本例中所创建的图表窗口的宽度为8*80 = 640像素。
    # plt.show()
    # tr=select_data(housing_data,0.2)
    # data_set(housing_data)
    # data_viual_show(housing_data)

    # 计算个特征之间的相关系数
    # coefficient(housing_data)

    # data_clean(housing_data)
    # atti_add=Attribute_Add()
    # housing_data_add=atti_add.fit_transform(housing_data.values)
    # housing_data.values转换成python中的普通numpy数组，如果需要转换成pandas中的dataframe，需要使用下列的代码
    # tp=pd.DataFrame(housing_data_add)
    # print(tp)
    """
    sklearn 中的流水线处理数字数据
    """
    num_attributes = list(housing_data.drop("ocean_proximity", axis=1))
    num_pipeline = Pipeline([
        ('get_num', get_data(num_attributes)),
        ('imputer', Imputer(strategy="median")),
        ('add_attributes', Attribute_Add()),
        ('normalize', StandardScaler()),

    ])
    """
    sklearn中的流水线处理非数字数据
    """
    cat = ["ocean_proximity"]
    N_num_pipeline = Pipeline([
        ('get_char', get_data(cat)),
        ('label_bin', MyOneHotEncode()),
    ])

    union_pipeline = FeatureUnion(transformer_list=[
        ("pipeline1", num_pipeline),
        ("pipeline2", N_num_pipeline),
    ])
    tq = union_pipeline.fit_transform(housing_data)
    # list=["rooms_per_househols","rooms_per_population","hot1","hot2","hot3","hot4","hot5"]
    # num_attributes.extend(list)
    # final_data=pd.DataFrame(tq,columns=num_attributes)
    # print(final_data.head())
    train_data = tq;
    train_lable = housing_data["median_house_value"]
    linearregression = LinearRegression()
    linearregression.fit(train_data, train_lable)

    dtr = DecisionTreeRegressor()
    dtr.fit(train_data, train_lable)

    test_data = housing_data.iloc[:14544]
    labels = housing_data["median_house_value"]
    test_label = labels.iloc[:14544]
    Ptest_data = union_pipeline.fit_transform(test_data)
    # 线性回归
    predict_data = linearregression.predict(Ptest_data)
    # 决策树
    # predict_data=dtr.predict(Ptest_data)

    # K交叉验证验证决策树
    scores1 = cross_val_score(dtr, train_data, train_lable, scoring="neg_mean_squared_error", cv=10)
    rmse1 = np.sqrt(-scores1)
    print("决策树十次交叉验证的评分:\t", rmse1)
    print("平均值：\t", rmse1.mean())
    print("方差：\t", rmse1.std())
    joblib.dump(dtr, "decession_tree")  # 保存我们训练后的模型以及相关的参数 加载使用joblib.load(dtr,""decession_tree)

    # K交叉验证验证线性回归
    scores2 = cross_val_score(linearregression, train_data, train_lable, scoring="neg_mean_squared_error", cv=10)
    rmse2 = np.sqrt(-scores2)
    print("线性回归十次交叉验证的评分:\t", rmse2)
    print("平均值：\t", rmse2.mean())
    print("方差：\t", rmse2.std())
    joblib.dump(linearregression, "line_regression")
    # 验证平均标准差
    line_mse = mean_squared_error(test_label, predict_data)
    line_rmse = np.sqrt(line_mse)
    # print(line_rmse)
    # print("the predict value:\t",predict_data) # 非格式化输出，不需要%分号
    # print("the true value:\t",test_label)
    # print("平均绝对偏差：\t",line_rmse)

    predict = linearregression.predict(train_data)
    line_mse = mean_squared_error(train_lable, predict)
    line_rmse = np.sqrt(line_mse)
    print("10次交叉验证训练后优化得到的模型：", line_rmse)
