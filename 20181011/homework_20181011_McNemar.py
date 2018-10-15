# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:17:35 2018

@author: Yi Tai
"""

import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('mlb_2017_regular_season_top_hitting.csv')
label_encoder = preprocessing.LabelEncoder()
df["Pos"] = label_encoder.fit_transform(df["Pos"]) #將字串數值化
#print(df.head(10))
df_data = df[["Pos","AVG?","OPS","OBP","SLG"]]
#df_data=pd.DataFrame(df,columns=['Pos','AVG?','OBP','OPS','SLG'])
df_target = df['2017AllStar']

# split train & test
train_ID3_X, test_ID3_X, train_ID3_y, test_ID3_y = train_test_split(df_data, df_target, test_size = 0.25)
train_CART_X, test_CART_X, train_CART_y, test_CART_y = train_test_split(df_data, df_target, test_size = 0.25)

# create decision tree CART
decision_tree = DecisionTreeClassifier(criterion='gini')
decision_tree_clf = decision_tree.fit(train_CART_X,train_CART_y)
predict_CART=decision_tree_clf.predict(test_CART_X)

# create decision tree ID3
decision_tree_id3 = DecisionTreeClassifier(criterion='entropy')
decision_tree_id3_clf = decision_tree.fit(train_ID3_X, train_ID3_y)
predict_ID3=decision_tree_clf.predict(test_ID3_X)
#比較表
voters=pd.DataFrame({"ID3":predict_ID3,"CART":predict_CART})
voter_tab=pd.crosstab(voters.ID3,voters.CART,margins=True)
observed=voter_tab.iloc[0:3,0:3]
print(observed,"\n")

#Z統計量
B=voter_tab.loc[0][1]
C=voter_tab.loc[1][0]
Z=(B-C)/(B+C)**0.5
Z_Critical=stats.norm.ppf(0.975) #大樣本Z分配
print("虛無假設:")
print("H0:ID3和CART效能一樣")
print("H1:ID3和CART效能不一樣")
print("McNemar檢定統計量:",Z)
print("α0.05 = > z0.05:",Z_Critical)
if Z > Z_Critical or Z<-(Z_Critical):
    print("結論顯著，ID3和CART效能不一樣")
else:
    print("結論不顯著，ID3和CART效能一樣")
        
    













