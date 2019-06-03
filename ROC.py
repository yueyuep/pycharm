from sklearn.metrics import roc_curve
import numpy as np
really_label= np.array([1, 1, 3, 2])
"""
really_label为实际的标签值，roc_curve计算的二分类的roc曲线，英雌我们可以假设某一个标签为正例
其他的为反例，
"""
predicts_score=np.array([0.1, 0.4, 0.35, 0.8])
"""
predicts_score为该列上的数据被预测成改列上所对应类的概率
"""

FTR,TPR,threshold=roc_curve(really_label,predicts_score,pos_label=2)
"""
threshold 为predicts_score的逆序排列
threshold=[0.8,0.4,0.35,0.1]
对应每个threshold 依次与predicts_score比较如果后者大与等于则为正例，反之为反例
predicts=[0,0,0,1]
TPR（真正率）=（1/(1+0)）=1
FNR（假阴率）=（0/（0+1））=0
因此理论上的值：
TPR[0]=0.25
FNR[0]=0.75
threshold=0.8
"""
print("FNR[0]:\tTPR[0]:\tthreshold:\t",FTR[0],TPR[0],threshold[0])