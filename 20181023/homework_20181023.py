# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:21:44 2018

@author: Yi Tai
"""
"""
作業:
    PCA:非監督式特徵轉換 LDA:監督式
    一、
    1.載入數據集
    2.描述性分析
    3.散佈圖(兩兩變數)
    4.相關矩陣
    5.共變異數矩陣
    6.eigenvalue分解
    7.找出主成分矩陣
    8.選出主成分比較不同數目主成分的MSE
    9.說明解釋量和eigenvalue和MSE的關係
    
    二、
    直接用scikit learn的PCA和回歸分析
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

#載入數據集
df = pd.read_csv('mlb_2017_regular_season_top_hitting.csv' ,index_col="Player")
#size_mapping = {"DH": 1.0,
#                "RF": 0.9,
#                "CF": 0.8,
#                "LF": 0.7,
#                "SS": 0.6,
#                "3B": 0.5,
#                "2B": 0.4,
#                "1B": 0.3,
#                "C": 0.2,
#                "P": 0.1}
#print(size_mapping)
#df['Pos'] = df['Pos'].map(size_mapping)
label_encoder = preprocessing.LabelEncoder()
df["Pos"] = label_encoder.fit_transform(df["Pos"])
#描述性統計
print(df.describe())

#散佈圖
cols=["Pos","AVG?","OBP","OPS"]
sns.pairplot(df[cols], size=2.5) #kind="reg",
plt.tight_layout()
plt.show()

#相關矩陣
np_corr=df[cols].corr()
print(np_corr)

#共變異數矩陣
df_data = df[["Pos","AVG?","OPS","OBP","SLG"]]
df_target = df["2017AllStar"]
train_X, test_X, train_y, test_y = train_test_split(df_data, df_target, test_size = 0.3, random_state=0)
sc=StandardScaler()
train_X_std=sc.fit_transform(train_X)
test_X_std=sc.fit_transform(test_X)
cov_mat=np.cov(train_X_std.T)
print(cov_mat)

#eigenvalue分解
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues特徵值 \n%s' % eigen_vals)
print('\nEigenvectors特徵向量 \n%s' % eigen_vecs)
tot=sum(eigen_vals)
var_exp=[(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp=np.cumsum(var_exp)
plt.bar(range(0,5),var_exp,alpha=0.5,align='center',label='individual explained variance')
plt.step(range(0,5),cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend()
plt.show()

eig_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eig_pairs.sort(reverse=True)

for i in range(5):
#    eig_pairs[i][1].sort(reverse=True)
    print(eig_pairs[i][1])

#主成分矩陣
w = np.hstack((eig_pairs[0][1][:,np.newaxis].real,eig_pairs[1][1][:,np.newaxis].real))
print('主成分矩陣Matrix W:\n',w)

#選出主成分比較不同數目主成分的MSE
#----
for i in range(5):
    a=i+1
    estimator = PCA(n_components=a)
    pca_x_train = estimator.fit_transform(train_X_std)
    pca_x_test = estimator.transform(test_X_std)
#降維
#pca_svc = LinearSVC()
#pca_svc.fit(pca_x_train, train_y)
#pca_y_predict = pca_svc.predict(pca_x_test)
#mse = np.mean((pca_y_predict - test_y) ** 2)
#print("Mean squared error",1,":",mse)

#----
    lm = LinearRegression()
#lm.fit(eig_pairs,eigen_vals)
    lm.fit(pca_x_train, train_y)
#mse = np.mean((lm.predict(dfops) - dfp) ** 2)
    pca_y_predict = lm.predict(pca_x_test)
    mse = np.mean((pca_y_predict - test_y) ** 2)
    print("Mean squared error",a,":",mse)

#說明解釋量和eigenvalue和MSE的關係
#print("利用Eigenvalue及解釋量來選取主成分個數\n，依照我的特徵值及解釋量會選擇適合主成分\n，選擇兩個主成分因為選擇兩個主成分已經\n使得資料有相當高的解釋量及特徵值，且每個\n維數的MSE都不太明顯，故直接選擇兩個主成\n分分析便能達到降低複雜度及預測目的。")
print("第一維度組:\nEigenvalue:",3.25198,"解釋量:",0.64389,"MSE:",0.14383)
print("第二維度組:\nEigenvalue:",0.95014,"解釋量:",0.83201,"MSE:",0.14397)
print("第三維度組:\nEigenvalue:",0.613385,"解釋量:",0.95346,"MSE:",0.14472)
print("第四維度組:\nEigenvalue:",0.234985,"解釋量:",0.99998,"MSE:",0.13695)
print("第五維度組:\nEigenvalue:",1.78728e-05,"解釋量:",1.00000,"MSE:",0.14740)
print("由各組主成分可知\n取一組主成分的特徵值最高，但是解釋量不足")
print("取兩組主成分的特徵值接近1，且解釋量足夠高")
print("取三組主成分的特徵值稍低，解釋量雖高，但是模型漸漸複雜")
print("取四組主成分的特徵值低，解釋量趨近1，但是維數過高、模型複雜")
print("取五組主成分的特徵值最低，可呈現完整解釋量，但是模型複雜、維數不變")
print("每組的MSE差異都不會過高，代表不同組數的主成分分析效果差不多")
print("為了降維使得模型簡化，且保有足夠解釋量的情形下，故選擇兩組主成分，來進行預測分析。")
