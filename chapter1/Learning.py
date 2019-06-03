import pandas as pd
from getdata import *
from select_data import *
from sklearn.linear_model import LinearRegression
housing_data=pd.read_csv("./chapter1/data/housing.csv")
processed_data=pd.read_csv("./chapter1/data/processed_data.csv")
labels=housing_data["median_house_value"]
liner_regression=LinearRegression()
liner_regression.fit(processed_data.drop('median_house_value',axis=1),labels)
test_lable=labels.iloc[:5]
test_data=union_pipeline()
liner_regression.fit(processed_data.drop('median_house_value',axis=1),labels)
predict=liner_regression.predict(Ptest_data)
print("the predict data:\t"%predict)
print("the true data: \t"%test_lable)


