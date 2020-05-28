# -*- coding: utf-8 -*-
"""
Created on Tue May 26 22:37:30 2020

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
import random
from pandas.core.common import SettingWithCopyWarning
from numpy.random import choice
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

data_folder ="data"

class read_required_data:
    def __init__(self,pred_dist,lastx):
        self.pdist = pred_dist
        self.lastx = lastx
    def read_dev_data(self):
        df_data = pd.read_csv("../data_for_learning_rank(all_distro_for_exp).csv",usecols=["person","distro","source"],encoding="utf-8")
        df_data.drop_duplicates(inplace=True)
        df_data = df_data[df_data["distro"]==self.pdist]
        return df_data
    
    def read_seq_sets_weights(self):
        wts = pd.read_table(os.path.join(data_folder,"pred_on_"+self.pdist+self.lastx+"_weights.txt"),names=["wt"])
        #print(wts)
        wts["wt"] = wts["wt"].astype('float')
        weight = list(wts["wt"])
        return weight

    def read_seq_data(self):
        seq_data = open(os.path.join(data_folder,"pred_on_"+self.pdist+self.lastx+"-seqs.txt"),"r")
        seq_dict = {}
        count = 1
        for line in seq_data:
            seq_dict[count] = line
            count +=1    
        return seq_dict
    
    def read_src_seq_lineNum(self):
        src_lineNum = pd.read_csv(os.path.join(data_folder,"data_dump_for_"+self.pdist+self.lastx+".csv"),names=["source","lineNum"])
        return src_lineNum

class main_process:
    def __init__(self,pred_dist,lastx):
        self.pred_dist = pred_dist
        self.lastx = lastx
    def load_data(self):
        rqrd_data = read_required_data(self.pred_dist)
        self.df_data = rqrd_data.read_dev_data()
        self.weights = rqrd_data.read_seq_sets_weights()
        self.seq_dict = rqrd_data.read_seq_data()
        self.srcLineNum = rqrd_data.read_src_seq_lineNum()
        
        train_srcList = list(self.srcLineNum["source"])
        pred_data = self.df_data[self.df_data["source"].isin(train_srcList)]
        pred_grped =pred_data.groupby("source")
        srcList = []
        count =[]
        for src, grp in pred_grped:
            count.append(grp.shape[0])
            srcList.append(src)
        self.pred_srcs = srcList
        self.pred_sizes = count
        
    def result_file(self):
        self.fwr = csv.writer(open(os.path.join(data_folder,"predicted_developer_"+self.pred_dist+self.lastx+".csv"),"w",newline=""))
        
    def procedure(self):
        self.result_file()
        for k in range(len(self.pred_srcs)):
            src = self.pred_srcs[k]
            size = self.pred_sizes[k]
            lineNum = list(self.srcLineNum.loc[self.srcLineNum["source"]==src]["lineNum"])[0]
            get_seq = self.seq_dict[lineNum]
            #print(get_seq)
            get_seq = get_seq.split(";")
            sizelist = get_seq[0].split(",")
            sizelist = [int(i) for i in sizelist]
            
            elemlist = get_seq[1].split(",")
            elemlist = [int(i) for i in elemlist]
            getTuple = []
            
            for sz in sizelist:
                elems = elemlist[0:sz]
                #print(elems)
                getTuple.append((sz,elems))
                elemlist = elemlist[sz:]
            
            #print(getTuple)
            #print(self.weights)
            self.weights.append(0)
            algo = setGen_algorithm(src,size,getTuple,self.weights)
            R = algo.setGen_using_CRU()
            #print(R)
            for dev in R:
                self.fwr.writerow([src,dev])
            #sys.exit()
class setGen_algorithm:
    def __init__(self,src,size,seq_set,wts):
        self.p =0.9
        self.src = src
        self.size = size
        self.seq_set = seq_set
        wts.append(0)
        self.weights = wts.reverse()
        
    def setGen_using_CRU(self):
        R = []
        tupleIds = [i for i in range(len(self.seq_set))]
        while len(R)<self.size:
            sampledID = choice(tupleIds, p=self.weights)
            sampled_set = self.seq_set[sampledID]
            T = []
            for ele in sampled_set[1]:
                rndNum= random.random()
                if rndNum>=self.p:
                    T.append(ele)
            T = list(set(T))
            if len(R)+len(T)>self.size:
                while len(R)<self.size:
                    y = choice(T)
                    R.append(y)
                    T.remove(y)
            else:
                R.extend(T)
        return R
                    
            
            
        
main = main_process("yakkety","10")
main.load_data()
main.procedure()
        
        
        
        