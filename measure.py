# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:44:12 2020

@author: Rimi
"""

import csv
#from datetime import datetime  
#import time
from unidecode import unidecode
import json,sys
import csv
import pandas as pd
import numpy as np
import os
import joblib
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


data_folder = "data"
try:
    os.mkdir(data_folder)
except:
    pass

class get_data:
    def __init__(self,pred_dist,trainDList):
        self.pdist = pred_dist
        self.tdList = trainDList
        
    def read_data(self):
        #,"closes","high","medium","low","label"
        df_data = pd.read_csv("../data_for_learning_rank(all_distro_for_exp).csv",usecols=["person","distro","source"],encoding="utf-8")
        df_data.drop_duplicates(inplace=True)
        df_data_trn = df_data[df_data["distro"].isin(self.tdList)]
        trn_persons = list(set(list(df_data_trn["person"])))
        
        df_data_pred = df_data[df_data["distro"]==self.pdist]
        #pred_GT_persons = list(df_data_pred["person"])
        return trn_persons,df_data_pred
        
    def read_single_dev_data(self):
        df_fc = pd.read_csv("../fact_check("+self.tdList[0]+"_to_"+self.tdList[-1]+").csv",names=["source","person"])
        return df_fc["source"]
    
    def read_seq_labels(self):
        dev_labels = pd.read_table(os.path.join(data_folder,"pred_on_"+self.pdist+str(lastx)+"-element-labels.txt"),names=["id","person"],sep=",",encoding="ISO-8859-1")
        dev_labels["id"] = dev_labels["id"].astype('int')
        return dev_labels
        
    def read_predicted_seq(self):
        res = pd.read_csv(os.path.join(data_folder,"predicted_developer_"+self.pdist+str(lastx)+".csv"),names=["source","id"])
        return res
        
    def filter_data_for_final(self):
        trn_p, df_data_pred = self.read_data()
        df_fc = list(self.read_single_dev_data())
        dev_labels = self.read_seq_labels()
        df_res = self.read_predicted_seq()
        print(df_res.shape,df_data_pred.shape)
        #remove singleton cases
        df_data_pred = df_data_pred[~df_data_pred["source"].isin(df_fc)]
        df_res = df_res[~df_res["source"].isin(df_fc)]
        print(df_res.shape,df_data_pred.shape)
        df_res_map1 = pd.merge(df_res,dev_labels,on="id",how="inner")
        print(df_res_map1.columns,df_res_map1.shape)
        
        df_res_map1.rename(columns={"person":"pred_person"},inplace=True)
        df_res_map2 = pd.merge(df_res_map1,df_data_pred,on="source",how="inner")
        df_res_map2.dropna(inplace=True)
        print(df_res_map2.shape,df_res_map2.columns)
        
        cnt_matched = df_res_map2[df_res_map2["pred_person"]==df_res_map2["person"]].shape[0]
        cnt_unmatched = df_res_map2.shape[0] - cnt_matched
        print("Fraction of matched: ", cnt_matched/df_res_map2.shape[0])
        print("Fraction of unmatched: ", cnt_unmatched/df_res_map2.shape[0])
        #print(df_res_map2[~df_res_map2["source"].isin(list(df_res_map1["source"]))])
        df_res_map2.to_csv(os.path.join(data_folder,"final_predicted_devs_"+self.pdist+str(lastx)+".csv"),index=False)
        

nonLTS =  ["warty","hoary","breezy","edgy","feisty","gutsy","intrepid","jaunty","karmic","maverick","natty","oneiric","quantal","raring","saucy","utopic","vivid","wily","yakkety","zesty"]
pred_distro = "yakkety"
lastx = 10
pred_index = nonLTS.index(pred_distro)
train_distlist = [d for d in nonLTS[pred_index-lastx:pred_index]]
print(train_distlist)
data = get_data(pred_distro,train_distlist)
data.filter_data_for_final()