# Impute short missing segments in HR data with KNN

import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import json
import statistics 

from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
from copy import deepcopy
from numpy import savetxt
from itertools import groupby


# global directories, file names and variables
HR_DIR = 'Preop_HR_data/'
STEP_DIR = 'Preop_Step_data/'
LABELS_CSV_FILENAME = 'ground_truth.csv'



# generate KNN model for KNN-HS
def generate_train_data(lines_hr, lines_step):
    train_x=[]
    train_y=[]

    for num in range(len(lines_hr)):
        line_data_hr = lines_hr[num].strip().split(',')
        line_data_hr = [int(x) if x!='-1' else np.nan for x in line_data_hr]
        line_data_step = lines_step[num].strip().split(',')
        line_data_step=[int(x) for x in line_data_step]

        # Slidng time window = 5
        for i in range(len(line_data_hr)-5):
            # check continuity
            if all(not np.isnan(data) for data in line_data_hr[i:i+6]):
                current_x=line_data_hr[i:i+5]+line_data_step[i:i+6]
                train_x.append(current_x)
                train_y.append(line_data_hr[i+5])
    
    neigh = KNeighborsRegressor(n_neighbors=10)
    neigh.fit(train_x, train_y)
    return neigh


# KNN-HS imputation
def KNN_impute_threshold(line_data_hr, line_data_step, neigh, segment_threshold):
    hr_data=line_data_hr[::]

    group_data=[]
    index=0
    for key, group in groupby(hr_data):
        len_group=len(list(group))
        group_data.append([index, key, len_group])
        index+=len_group

    for data in group_data:
        # If missing
        if data[1]==-1:
            # Impute if the missing length <= segment_threshold
            if data[2]<=segment_threshold:
                for i in range(data[0], data[0]+data[2]):
                    if (hr_data[i-5:i]!=[]) and (not -1 in hr_data[i-5:i]):
                        current_x=hr_data[i-5:i]+line_data_step[i-5:i+1]
                        hr_data_predict=neigh.predict([current_x])
                        hr_data[i]=hr_data_predict[0]

    hr_data=[str(int(x)) for x in hr_data]

    return hr_data


# Do KNN-HS imputation if the missing segment <= segment_threshold
def impute_hr_raw_data_knn_threshold (device, segment_threshold):
    ind = device['web_id']
    print("web_id:", ind)

    f_hr = open(HR_DIR+ind+'.csv','r')
    lines_hr = f_hr.readlines()
    f_hr.close()

    f_step = open(STEP_DIR+ind+'.csv','r')
    lines_step = f_step.readlines()
    f_step.close()

    fo = open(HR_DIR+ind+'_imputed.csv','w')
    neigh=generate_train_data(lines_hr, lines_step)

    num=0
    for line in lines_hr:
        line_data_hr = line.strip().split(',')
        line_data_hr = [int(x) for x in line_data_hr]
        line_data_step = lines_step[num].strip().split(',')
        line_data_step=[int(x) for x in line_data_step]

        # Imputation
        hr_impute=KNN_impute_threshold(line_data_hr, line_data_step, neigh, segment_threshold)
        imputed_line = ','.join(hr_impute)
        fo.write(imputed_line+'\n')
        num+=1
    fo.close()


if __name__ == '__main__':
    print("Start imputing the raw data (given a missing segment length threshold)")
    segment_threshold=10
    df = pd.read_csv(LABELS_CSV_FILENAME)
    for _, device in df.iterrows():
        impute_hr_raw_data_knn_threshold(device, segment_threshold)

