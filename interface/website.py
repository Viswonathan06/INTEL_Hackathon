from flask import Flask, render_template, request
import cv2

app = Flask(__name__)

@app.route('/')
def home():
    print("running")
    return render_template('./index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    # Get the uploaded video file from the form data
    video_file = request.files['video']

    # Read the video file using OpenCV
    cap = cv2.VideoCapture(video_file)

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
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': duration,
        'bitrate': bitrate
    }

    return render_template('result.html', results=results)

app.run(debug=True)