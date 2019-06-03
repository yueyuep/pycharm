from sklearn.datasets import fetch_mldata
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
mnist_data=fetch_mldata('mnist-original',data_home='./')
seed=np.random.permutation(60000)
x=mnist_data["data"]
y=mnist_data["target"]
train_data=(x[:60000])[seed]
train_label=(y[:60000])[seed]
parameter_list={"n_neighbors":[1,3],"weights":("uniform","distance")}
k_neighbors=KNeighborsClassifier(n_jobs=2)
grid_search=GridSearchCV(k_neighbors,parameter_list,cv=3,scoring="neg_mean_squared_error")
grid_search.fit(train_data,train_label)
print(grid_search.best_params_)


