import numpy as np

def DFA(data, win_length, order):
    N=len(data)
    n=int(np.floor(N/win_length))
    N_new=int(n*win_length)

    data=data[:N_new]
    #print(type(data))
    data=data-np.nanmean(data)
    y=np.zeros(N_new)

    loc_fluc=0
    num_fit=0

    for i in range(len(y)):
        y[i]=np.nansum(data[:i+1])

    for i in range(n):
        if np.sum(np.isnan(data[i*win_length:(i+1)*win_length]))>0:
            continue

        current_y=y[i*win_length:(i+1)*win_length]
        current_x=np.arange(i*win_length,(i+1)*win_length)

        coef=np.polyfit(current_x, current_y, order)
        pfit=np.poly1d(coef)
        fit_y=pfit(current_x)

        loc_fluc+=np.sum(np.square(current_y-fit_y))
        num_fit+=1

    if num_fit==0:
        return np.nan
    loc_fluc=np.sqrt(loc_fluc/(num_fit*win_length))
    #loc_fluc=np.sqrt(loc_fluc)
    return loc_fluc
