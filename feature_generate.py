# Generate daily features
# Standardization is used when computing daily features

import numpy as np
import pandas as pd
from datetime import datetime
import matlab.engine
from collections import defaultdict
import json
from copy import deepcopy
from numpy import savetxt
from numpy import loadtxt
import random
import glob
import os
from itertools import groupby
import DFA
from scipy.stats import skew, kurtosis

eng = matlab.engine.start_matlab()

# global directories, file names and variables
HR_DIR = 'Preop_HR_data/'
STEP_DIR = 'Preop_Step_data/'
SLEEP_DIR = 'Preop_Sleep_data/'
SLEEP_SUMMARY_DIR = 'Preop_Sleep_summary/'
SAVE_PATH = 'Preop_Input/'
LABELS_CSV_FILENAME = 'ground_truth.csv'

NUM_FEATURES = 35

def compute_feature_step(data_step, data_hr):
    num_ob=np.sum(~np.isnan(data_hr))
    
    # All missing
    if num_ob==0:
        return [np.nan]*6

    # daily steps
    total_step_nor=np.sum(data_step[~np.isnan(data_hr)])/num_ob
    # max step value
    step_max=np.max(data_step[~np.isnan(data_hr)])

    # sedentary and active bouts
    total_daily_sedentary_time=0
    daily_sedentary_bout_count=0
    total_daily_active_time=0
    daily_active_bout_count=0

    i=0
    # RAPIDS uses 10 for activity threshold
    while i<len(data_step):
        if data_step[i]<10 and (not np.isnan(data_hr[i])):
            daily_sedentary_bout_count+=1
            while i<len(data_step) and data_step[i]<10 and (not np.isnan(data_hr[i])):
                total_daily_sedentary_time+=1
                i+=1
        elif data_step[i]>=10 and (not np.isnan(data_hr[i])):
            daily_active_bout_count+=1
            while i<len(data_step) and data_step[i]>=10 and (not np.isnan(data_hr[i])):
                total_daily_active_time+=1
                i+=1
        else: 
            i+=1

    return total_step_nor, step_max, total_daily_sedentary_time/num_ob, daily_sedentary_bout_count/num_ob, total_daily_active_time/num_ob, daily_active_bout_count/num_ob


class parse_hr:
    def __init__(self):
        self.starting_date = ""
    def readLine(self,data_points):
        HR_E, HR_ro, HR_F, HR_LH = eng.compute_2nd_features_single(matlab.double(data_points),nargout=4)
        return HR_E, HR_ro, HR_F, HR_LH

class parse_raw_sleep:
    def __init__(self):
        self.starting_date = ""
    def readLine(self,data_points):
        F_n_sleep, skewness_sleep, kurtosis_sleep = eng.dfa_sleep_single(matlab.double(data_points),nargout=3)
        return F_n_sleep, skewness_sleep, kurtosis_sleep


def generate_feature (device, hr_threshold):
    ind = device['web_id']
    print("web_id:", ind)

    # Read data
    f_hr = open(HR_DIR+ind+'_imputed.csv','r')
    lines_hr = f_hr.readlines()
    f_hr.close()

    f_step = open(STEP_DIR+ind+'.csv','r')
    lines_step = f_step.readlines()
    f_step.close()

    data_length = len(lines_hr)
    # Save daily features
    data_matrix=np.zeros(shape=(data_length, NUM_FEATURES),dtype='float') 

    current_day = 0
    for line in lines_hr:
        line_data_hr = line.strip().split(',')
        line_data_hr = [int(x) if x!='-1' else np.nan for x in line_data_hr]
        line_data_step = lines_step[current_day].strip().split(',')
        line_data_step=[int(x) for x in line_data_step]

        if line_data_hr.count(np.nan)>1440*(1-hr_threshold):
            # Below the extraction threshold
            data_matrix[current_day][0:20]=np.nan
            current_day+=1
            continue

        # Compute step features
        total_step_nor, step_max, total_daily_sedentary_time_nor, daily_sedentary_bout_count_nor, total_daily_active_time_nor, daily_active_bout_count_nor = compute_feature_step(np.array(line_data_step[420:1441]), np.array(line_data_hr[420:1441]))
        
        if np.isnan(daily_sedentary_bout_count_nor):
            sedentary_per_bout = np.nan
        elif daily_sedentary_bout_count_nor==0:
            sedentary_per_bout=0
        else:
            sedentary_per_bout = total_daily_sedentary_time_nor / daily_sedentary_bout_count_nor

        if np.isnan(daily_active_bout_count_nor):
            active_per_bout = np.nan
        elif daily_active_bout_count_nor==0:
            active_per_bout=0
        else:
            active_per_bout = total_daily_active_time_nor / daily_active_bout_count_nor


        data_matrix[current_day][0] = total_step_nor
        data_matrix[current_day][1] = step_max

        data_matrix[current_day][2] = total_daily_sedentary_time_nor
        data_matrix[current_day][3] = daily_sedentary_bout_count_nor
        data_matrix[current_day][4] = sedentary_per_bout

        data_matrix[current_day][5] = total_daily_active_time_nor
        data_matrix[current_day][6] = daily_active_bout_count_nor
        data_matrix[current_day][7] = active_per_bout

        # Compute HR features
        data_matrix[current_day][8]= (DFA.DFA(line_data_hr,10,1))
        data_matrix[current_day][9]= (DFA.DFA(line_data_hr,20,1))
        data_matrix[current_day][10]= (DFA.DFA(line_data_hr,30,1))
        data_matrix[current_day][11]= (DFA.DFA(line_data_hr,40,1))
        data_matrix[current_day][12]= (DFA.DFA(line_data_hr,50,1))
        data_matrix[current_day][13]= (DFA.DFA(line_data_hr,60,1))

        data_matrix[current_day][14] = skew([data for data in line_data_hr if not np.isnan(data)])
        data_matrix[current_day][15] = kurtosis([data for data in line_data_hr if not np.isnan(data)])

        hr_obj = parse_hr()
        HR_E, HR_ro, HR_F, HR_LH = hr_obj.readLine(line_data_hr)
        data_matrix[current_day][16] = HR_E
        data_matrix[current_day][17] = HR_ro
        data_matrix[current_day][18] = HR_F
        data_matrix[current_day][19] = HR_LH

        current_day +=1


    # generate features based on Sleep
    try:
        current_day = 0
        with open(SLEEP_DIR+ind+'.csv') as f_sleep:
            lines  = f_sleep.readlines()
            for line in lines:
                raw_line_split = line.strip().split(',')
                raw_line_split = [data for data in raw_line_split if data!='']

                if all(data=='-1' for data in raw_line_split):
                    data_matrix[current_day][20:28]=np.nan
                else:
                    line_split=[float(data) for data in raw_line_split]
                    sleep_obj = parse_raw_sleep()
                    F_n_sleep, skewness_sleep, kurtosis_sleep = sleep_obj.readLine(line_split)

                    data_matrix[current_day][20] = F_n_sleep[0][0]
                    data_matrix[current_day][21] = F_n_sleep[1][0]
                    data_matrix[current_day][22] = F_n_sleep[2][0]
                    data_matrix[current_day][23] = F_n_sleep[3][0]
                    data_matrix[current_day][24] = F_n_sleep[4][0]
                    data_matrix[current_day][25] = F_n_sleep[5][0]
                    data_matrix[current_day][26] = skewness_sleep
                    data_matrix[current_day][27] = kurtosis_sleep
                current_day += 1
    except:
        # Sleep data are all missing for the whole study
        for j in range(data_length):
            data_matrix[j][20:28]=np.nan


    # Read from Sleep Summary
    current_day = 0
    try:
        sleep_summary_filename = SLEEP_SUMMARY_DIR+ind+'.csv'
        df = pd.read_csv(sleep_summary_filename)

        for _, line in df.iterrows():
            data_matrix[current_day][28] = line['main_AwakeCounts'] if line['main_AwakeCounts'] != -1 else np.nan
            data_matrix[current_day][29] = line['main_efficiency'] if line['main_efficiency'] != -1 else np.nan
            data_matrix[current_day][30] = line['main_minutesAsleep'] if line['main_minutesAsleep'] != -1 else np.nan
            data_matrix[current_day][31] = line['main_minutesAwake'] if line['main_minutesAwake'] != -1 else np.nan
            data_matrix[current_day][32] = line['main_minutesToFallAsleep'] if line['main_minutesToFallAsleep'] != -1 else np.nan
            data_matrix[current_day][33] = line['main_remCounts'] if line['main_remCounts'] != -1 else np.nan
            data_matrix[current_day][34] = line['main_timeInBed'] if line['main_timeInBed'] != -1 else np.nan
                    
            current_day += 1
    except:
        for j in range(data_length):
            data_matrix[j][28:35]=np.nan

    # Save to files
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    savetxt(SAVE_PATH+ind+'_fitbit_features.csv', data_matrix, delimiter=',' , fmt = '%.6f')


if __name__ == '__main__':
    hr_threshold=480/1440
    df = pd.read_csv(LABELS_CSV_FILENAME)
    for _, device in df.iterrows():
        generate_feature (device, hr_threshold)

