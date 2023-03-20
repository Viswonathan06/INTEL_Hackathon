# %%
import pandas as pd
import numpy as np
import os

def aggregate_facial(path):
    file = path.split('/')[-1]
    i = 0
    agg_data = []
    df = pd.read_csv(path)
    print(file)
    df.drop(df.columns[:4], inplace=True, axis=1)
    vid_deets = file.split('.')[0].split('_')
    print(vid_deets)
    vidname = vid_deets[1]
    # print(vidnum)
    vid_deets = [vidname]
    data = df.to_numpy()
    mean = list(np.mean(data, axis=0))
    std = list(np.std(data, axis = 0))
    vid_deets.extend(mean)
    vid_deets.extend(std)
    # print(i)
    cols = ['video']
    for i in range(408):
        cols.append(i)
    agg_data.append(vid_deets)
    print(agg_data)
    agg_df = pd.DataFrame(agg_data)
    agg_df.columns = cols
    agg_df.to_csv("./features/"+vidname+'_facial.csv')
    return agg_df

