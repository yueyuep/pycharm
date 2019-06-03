import pandas as pd
import numpy as np

if __name__ == '__main__':
    data = pd.read_csv("data1.csv")
    attribute = data.loc[1:, 1:5]
    labels = data.loc[1:, 6]
    # ndarry 与python的列表数据类型的不同之处在于ndarry中存储的数据类型必须是相同的，
    # ndaary支持 int、flot32、float64、字符串和浮点型数据。
    # 整数型-->浮点型-->字符串型 注意在数据的转换过程中可能造成精度的丢失
    attribute_n = np.array(attribute, dtype=np.float32)  # ndarry
    labels_n = np.array(labels, dtype=np.float32)  # ndarry

    attribute_l = attribute_n.tolist()  # list
    labels_l = labels_n.tolist()  # list
