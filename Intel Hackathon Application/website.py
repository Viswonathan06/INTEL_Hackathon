from flask import Flask, render_template, request
import cv2
import os, sys
sys.path.append('./Extracting_Features')
sys.path.append('./oneAPI_Tensorflow')
from Extract_All_Data import get_training_file
from training_prediction import predict_single_video, training
from get_accuracy import process_files, append_interview_value
app = Flask(__name__)

@app.route('/')
def home():
    print("running")
    return render_template('./index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    # Get the uploaded video file from the form data
    video_file = request.files['video']
    video_path = './video/'+video_file.filename
    # Read the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    video_name = video_path.split('/')[-1].split('.')[0]
    data = get_training_file(video_path)

    # %%
    print(predict_single_video(data, video_name))

    pred, interview = process_files()
    final_pred = append_interview_value(interview, pred)
    # print(final_pred.head())
    front_end_data = final_pred.to_dict()
    print(front_end_data)
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the frame rate of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Get the width and height of the video frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the duration of the video
    duration = total_frames / fps

    # Get the bitrate of the video
    bitrate = int(cap.get(cv2.CAP_PROP_BITRATE))

    # Release the video capture object
    cap.release()

    # Return the results as a dictionary
    results = {
        'AGREEABLENESS_Z':  front_end_data['AGREEABLENESS_Z'][0],
        'CONSCIENTIOUSNESS_Z': front_end_data['CONSCIENTIOUSNESS_Z'][0],
        'EXTRAVERSION_Z': front_end_data['EXTRAVERSION_Z'][0],
        'NEGATIVEEMOTIONALITY_Z': front_end_data['NEGATIVEEMOTIONALITY_Z'][0],
        'OPENMINDEDNESS_Z': front_end_data['OPENMINDEDNESS_Z'][0],
        'interview': front_end_data['interview'][0]
    }

    return render_template('result.html', results=results)

app.run(debug=True)