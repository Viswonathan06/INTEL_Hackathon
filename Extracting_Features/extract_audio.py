# %%
from moviepy.editor import *
import scipy.io.wavfile as wav
import pandas as pd
import numpy as np
import librosa


# %%
def extract_mfcc_features(video_path):
    aud_cols = ['video',        0,        1,        2,        3,        4,
              5,        6,        7,        8,        9,       10,       11,
             12,       13,       14,       15,       16,       17,       18,
             19,       20,       21,       22,       23,       24,       25,
             26,       27,       28,       29,       30,       31,       32,
             33,       34,       35,       36,       37,       38,       39,
             40,       41,       42,       43,       44,       45,       46,
             47,       48,       49,       50,       51,       52,       53,
             54,       55,       56,       57,       58,       59]
    folders = []
    new_df = pd.DataFrame(columns=aud_cols)

    audio_save_path = video_path.replace('video', 'video/audio')   
    audio_save_path = audio_save_path.replace('mov', 'wav')   
    try:
        os.mkdir('../video/audio')
    except:
        print('Exists!!')
    print(audio_save_path)    
    audioclip = AudioFileClip(video_path)
    audioclip.write_audiofile(audio_save_path)
    x , sr = librosa.load(audio_save_path, sr = 44100)
    print(x, sr)
    mfccs = librosa.feature.mfcc(y=x,sr=sr)
    print(mfccs.shape)


    index = []
    sum = np.sum(mfccs, axis = 1)
    sorted_sum = np.sort(sum)[::-1]
    l = list(sum)
    for s in sorted_sum:
        index.append(l.index(s))
    std = np.std(mfccs, axis = 1)
    med = np.median(mfccs, axis = 1)
    index.extend(list(std))
    index.extend(list(med))
    data = []
    name = video_path.split('/')[2].split('.')[0]
    print(name)
    vid_details = [name]
    vid_details.extend(index)
    data.append(vid_details)
    audio_feats = 60
    cols = []
    for i in range(audio_feats):
        cols.append(i)
    columns = ['video']
    columns.extend(cols)
    temp = pd.DataFrame(columns=columns, data=data)
    new_df = new_df.append(temp, ignore_index=True)
    new_df.to_csv('../features/'+name+'_audio.csv')
    return new_df



            
            


