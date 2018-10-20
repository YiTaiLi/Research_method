# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 13:42:41 2018

@author: Yi Tai
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing  import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#d=np.random.randint(1,10,size=(4,4))
#print(d)
#b=np.linalg.inv(d) #inv可以用於計算反矩陣
#print(b)
#e=np.dot(d,b)
#print(e)
#f=np.dot(b,d)
#print(f)
#h=np.linalg.det(d) #欲計算矩陣行列式，可用det指令
#print(h)
#g=h*b
#print(g)
#m,n=np.linalg.eig(d)
#print(m,n)

#eigenvalue decomposition
# =>SVD
# =>PCA
#主成分分析
#矩陣相乘 =>坐標系轉換
#         =>向量平移旋轉
#a=[[1,2,3],[4,5,6]]
#p,q,r=np.linalg.svd(a)
#print(p)
#print(q)
#print(r)

"""
1.下載資料集
2.資料切割成訓練、測試
3.資料標準化
4.建立共變異矩陣
5.對共變異矩陣(方陣)做eigenvalue decomposition => eigenvector
6.讓資料轉換到新座標軸
sklearn畫圖

"""

df = pd.read_csv('mlb_2017_regular_season_top_hitting.csv')
label_encoder = preprocessing.LabelEncoder()
df["Pos"] = label_encoder.fit_transform(df["Pos"]) #將字串數值化
df_data = df[["Pos","AVG?","OPS","OBP","SLG"]]
df_target = df["2017AllStar"]
train_X, test_X, train_y, test_y = train_test_split(df_data, df_target, test_size = 0.3, random_state=0)
sc=StandardScaler()
train_X_std=sc.fit_transform(train_X)
test_X_std=sc.fit_transform(test_X)
cov_mat=np.cov(train_X_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)
tot=sum(eigen_vals)
var_exp=[(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp=np.cumsum(var_exp)
plt.bar(range(0,5),var_exp,alpha=0.5,align='center',label='individual explained variance')
plt.step(range(0,5),cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend()
plt.show()

#PCA
pca=PCA()
train_X_pca=pca.fit_transform(train_X_std)
print(pca.explained_variance_ratio_)
plt.bar(range(0,5),pca.explained_variance_ratio_,alpha=0.5,align='center')
plt.step(range(0,5),np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()
