import numpy as np
import pandas as pd
import glob
from datetime import datetime
import os

# Extract pre-op data before "dos"  (surgery date)
# Transpose pre-op data
# If the pre-op length is larger than 30, use the latest observed 30-day data


LABELS_CSV_FILENAME = "ground_truth.csv"
PREOP_HR_DIR = 'Preop_HR_data/'
PREOP_STEP_DIR = 'Preop_Step_data/'
PREOP_SLEEP_DIR = 'Preop_Sleep_data/'

# Some users did not wear the device at the beginning and ending of preop stage
# Use this funtion to find those missing dates
def find_missing_index(all_preop_hr_data):  
    missing_index=[]  
    
    for i in range(len(all_preop_hr_data)):
        if sum(all_preop_hr_data[i]) == -1*len(all_preop_hr_data[i]):  # All -1
            missing_index.append(i)
        else:
            break

    for i in range(len(all_preop_hr_data)-1, -1, -1):
        if sum(all_preop_hr_data[i]) == -1*len(all_preop_hr_data[i]): # All -1
            missing_index.append(i)
        else:
            break

    return list(set(missing_index))


if __name__ == "__main__":
    df_label = pd.read_csv(LABELS_CSV_FILENAME)

    for _, device in df_label.iterrows():
        ind = device["web_id"]
        surgery_date=device["dos"]

        # Read HR data
        path='./fitbit_data/'+ind+'/'
        hr_filename = glob.glob(path+'*intradayHeart.csv')[0]
        df_hr=pd.read_csv(hr_filename, sep=',')

        all_preop_hr_data = []
        preop_index = []
        i = 0
        for date_name in df_hr.columns[1:]:
            if date_name < surgery_date:
                current_hr_data = list (df_hr[date_name])
                all_preop_hr_data.append(current_hr_data)
                preop_index.append(i)
                i+=1
            else:
                break

        hr_transposed_array = np.array(all_preop_hr_data)
        missing_index = find_missing_index(all_preop_hr_data)

        # Remove the completely missing days at the start/end of the preop stage
        hr_transposed_array = np.delete(hr_transposed_array, missing_index, axis = 0)
        # If longer than 30 days, only use the latest observed 30-day data
        if len(hr_transposed_array)>30:
            hr_transposed_array = hr_transposed_array[-30:]

        # Save the extracted data
        hr_save_path = PREOP_HR_DIR 
        if not os.path.exists(hr_save_path):
            os.makedirs(hr_save_path)
        np.savetxt(hr_save_path + ind+'.csv', hr_transposed_array, fmt='%i', delimiter=",")


        # Read Step data
        step_filename = glob.glob(path+'*Step.csv')[0]
        df_step=pd.read_csv(step_filename, sep=',')

        all_preop_step_data = []
        for date_name in df_step.columns[1:]:
            if date_name < surgery_date:
                current_step_data = list (df_step[date_name])
                all_preop_step_data.append(current_step_data)
            else:
                break

        step_transposed_array = np.array(all_preop_step_data)

        # Remove the completely missing days at the start/end of the preop stage
        step_transposed_array = np.delete(step_transposed_array, missing_index, axis = 0)
        # If longer than 30 days, only use the latest observed 30-day data
        if len(step_transposed_array)>30:
            step_transposed_array = step_transposed_array[-30:]

        # Save the extracted data
        step_save_path = PREOP_STEP_DIR 
        if not os.path.exists(step_save_path):
            os.makedirs(step_save_path)
        np.savetxt(step_save_path + ind+'.csv', step_transposed_array, fmt='%i', delimiter=",")


        # Read Sleep data
        # Some users may not have sleep data for the whole study
        if glob.glob(path+'*sleep.csv'):
            sleep_filename = glob.glob(path+'*sleep.csv')[0]
            
            all_preop_sleep_data = []

            f = open(sleep_filename,'r')
            lines = f.readlines()
            f.close()

            for day_index in preop_index:
                line = lines[day_index]
                if day_index in missing_index:
                    continue
                current_sleep_data = [int(float(i)) for i in line.strip().split(',') if i!='']
                all_preop_sleep_data.append(current_sleep_data)

            # Save sleep data
            # Sleep data have different lengths per night
            pd_sleep = pd.DataFrame(all_preop_sleep_data)
            sleep_save_path = PREOP_SLEEP_DIR 
            if not os.path.exists(sleep_save_path):
                os.makedirs(sleep_save_path)

            pd_sleep.to_csv(sleep_save_path+ind+'.csv', index=False, header=False)

