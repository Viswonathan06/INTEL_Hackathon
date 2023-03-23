# %%
import scipy.io.wavfile as wav
import pandas as pd
import numpy as np
import os, sys
from multiprocessing import Process
from combine_features import aggregate_facial
# %%
def run_face_landmarks(video_path):
    print(video_path)
    save_path = '../features/NAH_mov_facial.csv'
    os.system('python3 ./demo_video_smooth.py -f'+video_path+' --onnx')  
    aggregate_data = aggregate_facial(save_path)
    return aggregate_data

# Run for multiple videos parallelly

# # %%
# if __name__ == '__main__':
#     main_folders = []
#     for fols in os.listdir(training_path):
#         if '.DS_Store' in fols or 'Audio' in fols or fols not in vish_folders:
#             continue
#         print(fols)
#         for files in os.listdir(training_path+'/'+fols):
#             if '.DS_Store' in files:
#                 continue
#             main_folders.append(training_path+'/'+fols+'/'+files)
#     fol_sep = np.array_split(main_folders,4)
#     print(fol_sep)
#     process_pool = []
#     for i in range(4):
#         t1 = Process(target=run_face_landmarks, args=(fol_sep[i],))
#         t1.daemon = True
#         t1.start()
#         process_pool.append(t1)

#     for t in process_pool:
#         t.join()



