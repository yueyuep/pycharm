import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn as skl
def prepare_country_stats(data1,data2):
    data2.rename(columns={"2015": "GDP_data"},inplace=True)
    data2.set_index("Country",inplace=True)
    full_data=pd.merge(left=data1,right=data2,left_index=True,right_index=True)
    full_data.sort_values(by="GDP_data",inplace=True)
    return  full_data

data1=pd.read_csv("oecd_bli_2015.csv",thousands=',')
data1=data1[data1["INEQUALITY"]=="TOT"]
data1=data1.pivot(index="Country",columns="Indicator",values="Value")
data2=pd.read_csv("gdp_per_capita.csv",thousands=',',delimiter='\t',encoding='latin1',na_values="n/a")
temp=prepare_country_stats(data1,data2)
x=np.c_[temp["GDP_data"]]
y=np.c_[temp["Life satisfaction"]]
plt.plot(x,y)





