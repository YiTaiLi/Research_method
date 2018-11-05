# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:34:43 2018

@author: Yi Tai
"""

"""
PCA降維
回歸用PCA
類別用LDA -> scikitlearn
"""

import pandas as pd
from sklearn import preprocessing, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from sklearn.linear_model  import LogisticRegression


df = pd.read_csv('mlb_2017_regular_season_top_hitting.csv' )
label_encoder = preprocessing.LabelEncoder()
df["Pos"] = label_encoder.fit_transform(df["Pos"])
df_data = df[["Pos","AVG?","OPS","OBP","SLG","RBI","SB","SO"]]

df_target = df["2017AllStar"]
X_train, X_test, y_train, y_test = train_test_split(
    df_data, df_target, test_size=0.3, random_state=0)

sc=StandardScaler()
train_X_std=sc.fit_transform(X_train)
test_X_std=sc.fit_transform(X_test)
#未降維
lr = LogisticRegression()
lr=lr.fit(train_X_std, y_train)
predict_y_lr = lr.predict(test_X_std)
#print(survived_predictions)
#print(np.array(df_target))
accuracylr = metrics.accuracy_score(y_test, predict_y_lr)
#accuracy = logistic_regr.score(df_data, df_target)
print("LDA降維前:\n邏輯回歸正確率:",accuracylr)
#LDA降維
#lda_x_train, lda_x_test = [], []

lda = LinearDiscriminantAnalysis(n_components=2)
lda_x_train = lda.fit_transform(train_X_std,y_train)
lda_x_test = lda.transform(test_X_std)
        
lrlda = LogisticRegression()
lrlda = lrlda.fit(lda_x_train,y_train)
predict_y_lda=lrlda.predict(lda_x_test)
#print(predict_y_lda)
accuracylda = metrics.accuracy_score(y_test, predict_y_lda)

print("LDA降維後:\n邏輯回歸正確率:",accuracylda)

