# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 19:43:33 2018

@author: Yi Tai
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

df = pd.read_csv('mlb_2017_regular_season_top_hitting.csv' ,index_col="Player")
label_encoder = preprocessing.LabelEncoder()
df["Pos"] = label_encoder.fit_transform(df["Pos"])
df_data = df[["Pos","AVG?","OPS","OBP","SLG"]]
df_target = df["2017AllStar"]
train_X, test_X, train_y, test_y = train_test_split(df_data, df_target, test_size = 0.3, random_state=0)
sc=StandardScaler()
train_X_std=sc.fit_transform(train_X)
test_X_std=sc.fit_transform(test_X)
cov_mat=np.cov(train_X_std.T)

estimator = PCA(n_components=2)
pca_x_train = estimator.fit_transform(train_X_std)
pca_x_test = estimator.transform(test_X_std)
lm = LinearRegression()
lm.fit(pca_x_train, train_y)
pca_y_predict = lm.predict(pca_x_test)
mse = np.mean((pca_y_predict - test_y) ** 2)
print("Mean squared error:",mse)
print('Slope: %.3f' % lm.coef_[0])
print('Intercept: %.3f' % lm.intercept_)