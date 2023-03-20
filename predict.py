# %%
import os, sys
sys.path.append('./Extracting_Features')
sys.path.append('./oneAPI_Tensorflow')
from Extract_All_Data import get_training_file
from training_prediction import predict_single_video, training

# %%
# os.chdir("../Feature_Extraction/")


video_path = './video/NAH.mov'
video_name = video_path.split('/')[-1].split('.')[0]
data = get_training_file(video_path)

# %%
data.head()

# %%
predict_single_video(data, video_name)


