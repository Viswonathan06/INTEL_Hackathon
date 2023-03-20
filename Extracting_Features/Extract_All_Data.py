# %%
import sys, os
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/viswonathanmanoranjan/audio-orchestrator-ffmpeg/bin/ffmpeg"
# sys.path.append('./3DDFA_V2')
from extract_audio import extract_mfcc_features
from combine_features import aggregate_facial
import pandas as pd



# %%
def run_face_landmarks(video_path):
    print(video_path)
    save_path = './features/aggregate_'+video_path.split('/')[-1].split('.')[0]+'_facial.csv'
    print(os.system('ls -s'))    
    os.system('python3 ./demo_video_smooth.py -f'+video_path+' --onnx')  
    print(save_path)
    aggregate_data = aggregate_facial(save_path)
    return aggregate_data

def get_training_file(video_path):
    mfcc = extract_mfcc_features(video_path)
    facial = run_face_landmarks(video_path)
    facial = facial.add_suffix('_faci')
    facial = facial.rename(columns={'video_faci': 'video'})
    mfcc = mfcc.add_suffix('_audi')
    mfcc = mfcc.rename(columns={'video_audi': 'video'})
    base_df = facial.merge(mfcc,on=['video'], suffixes=('', '_audi'))
    base_df.to_csv("./features/combined/training_"+video_path.split('/')[-1].split('.')[0]+'.csv')
    return base_df



