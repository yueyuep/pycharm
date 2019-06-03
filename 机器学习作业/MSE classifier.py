from sklearn.linear_model import LinearRegression as lr
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
data = pd.read_csv(r"data1.csv")
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
#划分训练数据和测试数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
#标准化数据

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

#创建模型
model = lr()
model.fit(x_train_std, y_train)
y_pred = model.predict(x_test_std)
res = []

for i in y_pred:
    if i > 0.5:
        res.append(1)
    else:
        res.append(0)
np.array(res)
np.array(y_test)

print ('Accuracy:%.2f' % accuracy_score(y_test,res))
print('错误分类数: %d' % (y_test != res).sum())