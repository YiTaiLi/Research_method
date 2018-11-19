# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 21:13:22 2018

@author: Yi Tai
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


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

#---------------
gnb = GaussianNB()
gnb.fit(train_X_std, y_train)
y_pred=gnb.predict(test_X_std)
print(y_pred)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print("正確率:",accuracy)