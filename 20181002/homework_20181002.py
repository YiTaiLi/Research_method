# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:57:12 2018

@author: Yi Tai
"""
#區間估計--------------------------------
import numpy as np
from scipy import stats
import math
import pandas as pd
from scipy.stats import ttest_ind


df= pd.read_csv('mlb_2017_regular_season_top_hitting.csv')
df_ops = df["OPS"]
df_avg = df["AVG?"]
df_obp = df["OBP"]
df_as = df["2017AllStar"]

print("OPS母體平均:", sum(df_ops)/144)
print("AVG母體平均:", sum(df_avg)/144)
print("OBP母體平均:", sum(df_obp)/144)


#print(df[df['2017AllStar']==1]['OPS'].describe())
#print(df['OPS'].describe())
#print(df[df['2017AllStar']==0]['OPS'].describe())
#print(df[df['2017Si.S']==1]['OPS'].describe())
#大樣本區間估計--------------- 
sample_size = 60
sample_ops = np.random.choice(df_ops, sample_size)
sample_avg = np.random.choice(df_avg, sample_size)
sample_obp = np.random.choice(df_obp, sample_size)
sample_as = np.random.choice(df_as, sample_size)

sample_ops_mean = sample_ops.mean()
sample_avg_mean = sample_avg.mean()
sample_obp_mean = sample_obp.mean()
sample_as_mean = sample_as.mean()
print("OPS樣本平均:", sample_ops_mean)
print("AVG樣本平均:", sample_avg_mean)
print("OBP樣本平均:", sample_obp_mean)
print("AllStar樣本平均:", sample_as_mean)

sample_ops_stdev = sample_ops.std()
sample_avg_stdev = sample_avg.std()
sample_obp_stdev = sample_obp.std()
sample_as_stdev = sample_as.std()
print("OPS樣本標準差:", sample_ops_stdev)
print("AVG樣本標準差:", sample_avg_stdev)
print("OBP樣本標準差:", sample_obp_stdev)
print("AllStar樣本標準差:", sample_as_stdev)

sigma_ops = sample_ops_stdev/math.sqrt(sample_size-1)
sigma_avg = sample_avg_stdev/math.sqrt(sample_size-1)
sigma_obp = sample_obp_stdev/math.sqrt(sample_size-1)
sigma_as = sample_as_stdev/math.sqrt(sample_size-1)
print("OPS樣本計算出的母體標準差:", sigma_ops)
print("AVG樣本計算出的母體標準差:", sigma_avg)
print("OBP樣本計算出的母體標準差:", sigma_obp)
print("AllStar樣本計算出的母體標準差:", sigma_as)

z_critical = stats.norm.ppf(q=0.975)
print("z分數:", z_critical)

#margin_of_error_ops = z_critical * sigma_ops
#confidence_interval_ops = (sample_ops_mean - margin_of_error_ops,
#                       sample_ops_mean + margin_of_error_ops)
#margin_of_error_avg = z_critical * sigma_avg
#confidence_interval_avg = (sample_avg_mean - margin_of_error_avg,
#                       sample_avg_mean + margin_of_error_avg)
#margin_of_error_obp = z_critical * sigma_obp
#confidence_interval_obp = (sample_obp_mean - margin_of_error_obp,
#                       sample_obp_mean + margin_of_error_obp)
#margin_of_error_as = z_critical * sigma_as
#confidence_interval_as = (sample_as_mean - margin_of_error_as,
#                       sample_as_mean + margin_of_error_as)
#
#print(confidence_interval_ops)
#print(confidence_interval_avg)
#print(confidence_interval_obp)
#print(confidence_interval_as)

conf_int_ops = stats.norm.interval(alpha=0.95,                 
                               loc=sample_ops_mean, 
                               scale=sigma_ops)
conf_int_avg = stats.norm.interval(alpha=0.95,                 
                               loc=sample_avg_mean, 
                               scale=sigma_avg)
conf_int_obp = stats.norm.interval(alpha=0.95,                 
                               loc=sample_obp_mean, 
                               scale=sigma_obp)
conf_int_as = stats.norm.interval(alpha=0.95,                 
                               loc=sample_as_mean, 
                               scale=sigma_as)
print(conf_int_ops[0],'~', conf_int_ops[1])
print(conf_int_avg[0],'~',conf_int_avg[1])
print(conf_int_obp[0],'~', conf_int_obp[1])
print(conf_int_as[0],'~', conf_int_as[1])
#小樣本區間估計-----------------
sample_sizet = 20
sample_opst = np.random.choice(df_ops, sample_sizet)
sample_avgt = np.random.choice(df_avg, sample_sizet)
sample_obpt = np.random.choice(df_obp, sample_sizet)
sample_ast = np.random.choice(df_as, sample_sizet)

sample_opst_mean = sample_opst.mean()
sample_avgt_mean = sample_avgt.mean()
sample_obpt_mean = sample_obpt.mean()
sample_ast_mean = sample_ast.mean()
print("OPS小樣本平均:", sample_opst_mean)
print("AVG小樣本平均:", sample_avgt_mean)
print("OBP小樣本平均:", sample_obpt_mean)
print("AllStar小樣本平均:", sample_ast_mean)

sample_opst_stdev = sample_opst.std()
sample_avgt_stdev = sample_avgt.std()
sample_obpt_stdev = sample_obpt.std()
sample_ast_stdev = sample_ast.std()
print("OPS小樣本標準差:", sample_opst_stdev)
print("AVG小樣本標準差:", sample_avgt_stdev)
print("OBP小樣本標準差:", sample_obpt_stdev)
print("AllStar小樣本標準差:", sample_ast_stdev)

sigma_opst = sample_opst_stdev/math.sqrt(sample_sizet-1)
sigma_avgt = sample_avgt_stdev/math.sqrt(sample_sizet-1)
sigma_obpt = sample_obpt_stdev/math.sqrt(sample_sizet-1)
sigma_ast = sample_ast_stdev/math.sqrt(sample_sizet-1)
print("OPS小樣本計算出的母體標準差:", sigma_opst)
print("AVG小樣本計算出的母體標準差:", sigma_avgt)
print("OBP小樣本計算出的母體標準差:", sigma_obpt)
print("AllStar小樣本計算出的母體標準差:", sigma_ast)

t_critical = stats.t.ppf(q=0.975,df=sample_sizet-1)
print("t分數:", t_critical)

conf_int_opst = stats.t.interval(alpha=0.95,
                            df=sample_sizet-1,
                            loc=sample_opst_mean, 
                            scale=sigma_opst)
conf_int_avgt = stats.t.interval(alpha=0.95,
                            df=sample_sizet-1,
                            loc=sample_avgt_mean, 
                            scale=sigma_avgt)
conf_int_obpt = stats.t.interval(alpha=0.95,
                            df=sample_sizet-1,
                            loc=sample_obpt_mean, 
                            scale=sigma_obpt)
conf_int_ast = stats.t.interval(alpha=0.95,
                            df=sample_sizet-1,
                            loc=sample_ast_mean, 
                            scale=sigma_ast)
print(conf_int_opst[0],'~', conf_int_opst[1])
print(conf_int_avgt[0],'~',conf_int_avgt[1])
print(conf_int_obpt[0],'~', conf_int_obpt[1])
print(conf_int_ast[0],'~', conf_int_ast[1])

#T檢定----------------------
sample_opst_size=len(sample_opst)
sample_avgt_size=len(sample_avgt)
sample_obpt_size=len(sample_obpt)
sample_ast_size=len(sample_ast)

population_opst_mean = 0.79 #建立OPS假設
population_avgt_mean = 0.27 #建立AVG假設
population_obpt_mean = 0.34 #建立OBP假設

t_obtained_ops = (sample_opst_mean-population_opst_mean)/sigma_opst
t_obtained_avg = (sample_avgt_mean-population_avgt_mean)/sigma_avgt
t_obtained_obp = (sample_obpt_mean-population_obpt_mean)/sigma_obpt
print("OPS-T檢定統計量:",t_obtained_ops)
print("AVG-T檢定統計量:",t_obtained_avg)
print("OBP-T檢定統計量:",t_obtained_obp)
t_critical_ops=stats.t.ppf(q=0.975,df=sample_opst_size-1)
print("T分數",t_critical_ops)

#獨立樣本t檢定
df_asc=df[df["2017AllStar"]==1][['AVG?','OBP','OPS']]
df_asn=df[df["2017AllStar"]==0][['AVG?','OBP','OPS']]
print(df_asc.describe())
print(df_asn.describe())
#df_asc=df[df["League"]==1][['AVG?','OBP','OPS']]
#df_asn=df[df["League"]==0][['AVG?','OBP','OPS']]
df_as_ttest = ttest_ind(df_asc,df_asn)
print(df_as_ttest)

#獨立樣本t檢定函式
def t_test(group1, group2):
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    std1 = np.std(group1)
    std2 = np.std(group2)
    nobs1 = len(group1)
    nobs2 = len(group2)
    
    modified_std1 = np.sqrt(np.float32(nobs1)/np.float32(nobs1-1)) * std1
    modified_std2 = np.sqrt(np.float32(nobs2)/np.float32(nobs2-1)) * std2
    (statistic, pvalue) = stats.ttest_ind_from_stats(mean1=mean1, std1=modified_std1, nobs1=nobs1, mean2=mean2, std2=modified_std2, nobs2=nobs2)
    return statistic, pvalue
#print(t_test(df_asc,df_asn))

#隊伍分類值    
#size_mapping = {"DH": 10,
#                "RF": 9,
#                "CF": 8,
#                "LF": 7,
#                "SS": 6,
#                "3B": 5,
#                "2B": 4,
#                "1B": 3,
#                "C": 2,
#                "P": 1}
#print(size_mapping)
#df['Pos'] = df['Pos'].map(size_mapping)
#print(df.head(10))
