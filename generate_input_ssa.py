import numpy as np
import pandas as pd
from numpy import loadtxt
import warnings
from sklearn import linear_model
from scipy.linalg import toeplitz
from numpy import savetxt


DAILY_FEATURE_DIR = 'Preop_Input/'
LABELS_CSV_FILENAME = 'ground_truth.csv'
NUM_NON_SLEEP_FEATURES = 20


# Embed time series data into Toeplitz lagged correlation matrix
# x: time series data
# N: time series data length
# L: window length
def time_to_C(x, N, L):
    # c: diagonal in C
    c=[]
    for j in range(L):
        c_current=[]
        for i in range(N-j):
            if ~np.isnan(x[i]) and ~np.isnan(x[i+j]):
                c_current.append(x[i]*x[i+j])
        c.append(np.mean(c_current))
    
    #print(c)
    C= toeplitz(c)
    #print(C)
    return C

# Extract trend with SSA
def extract_trend(x, N, L):
    C=time_to_C(x, N, L)
    #svd
    U, Sigma, V = np.linalg.svd(C)
    
    # first principal component
    a=[]
    for i in range(1, N-L+2):
        a_current=[]
        for j in range(1, L+1):
            if ~np.isnan(x[i+j-2]):
                a_current.append(x[i+j-2]*U[j-1][0])
        a.append(np.mean(a_current)*L)
    
    # extract trend
    # reconstruct with the first principal component
    x_re=[]
    for i in range(1, L):
        current_re=0
        for j in range(1, i+1):
            current_re+=a[i-j]*U[j-1][0]
        x_re.append(current_re/i)
    
    for i in range(L, N-L+2):
        current_re=0
        for j in range(1,L+1):
            current_re+=a[i-j]*U[j-1][0]
        x_re.append(current_re/L)
    
    
    for i in range(N-L+2, N+1):
        current_re=0
        for j in range(i-N+L,L+1):
            current_re+=a[i-j]*U[j-1][0]
        x_re.append(current_re/(N-i+1))
    
    return x_re


# Delete the missing data at the head/tail
# Find the index of these missing data
def find_delete_index(data):  
    delete_index=[]  
    
    for i in range(len(data)):
        if np.isnan(data[i]):
            delete_index.append(i)
        else:
            break

    for i in range(len(data)-1, -1, -1):
        if np.isnan(data[i]):
            delete_index.append(i)
        else:
            break

    return list(set(delete_index))


# Compute slope with missing data
def cal_trend_with_nan(data):
    time_list = np.arange(len(data))
    # create linear regression object
    mask = ~np.isnan(data)
    regr = linear_model.LinearRegression()
    regr.fit(time_list.reshape(-1,1)[mask], np.array(data).reshape(-1,1)[mask])

    return regr.coef_[0][0]


# Generate the input vector from data_matrix
# Use SSA to generate high-level features
def create_high_level_features(data_matrix):
    num_features = data_matrix.shape[1]
    delete_index_hr=find_delete_index(data_matrix[:,0])
    delete_index_sleep=find_delete_index(data_matrix[:,-1])
    
    feature_array = [0.0 for i in range(3*num_features)]
    
    for fi in range(num_features):
        if fi<NUM_NON_SLEEP_FEATURES:
            delete_index=delete_index_hr
        else:
            delete_index=delete_index_sleep

        # Delete the missing data at the begining and ending of the study
        data=np.delete(data_matrix[:,fi], delete_index)
    
        N=len(data)
        L=3
        try:
            trend=extract_trend(data, N, L)
            feature_array[fi] = cal_trend_with_nan(trend)
            feature_array[num_features+fi] = np.nanmean(trend)
            feature_array[2*num_features+fi] = np.nanvar(trend)
        except:
            # Many users have missing sleep features
            # SSA is not working if the data were missing for most of the days
            # In our case, some users had no sleep data 
            feature_array[fi] = np.nan
            feature_array[num_features+fi] = np.nan
            feature_array[2*num_features+fi] = np.nan

    return feature_array


if __name__ == '__main__':
    fitbit_dataset = []
    df_label = pd.read_csv(LABELS_CSV_FILENAME)

    for _, device in df_label.iterrows():
        ind = device['web_id']

        # Load fitbit features
        data_matrix = loadtxt(DAILY_FEATURE_DIR+ind+'_fitbit_features.csv', delimiter=',')
        fitbit_feature_vector = create_high_level_features(data_matrix)

        savetxt(DAILY_FEATURE_DIR+ ind +'_fitbit_vector.csv', fitbit_feature_vector, delimiter=',' , fmt = '%.6f')


