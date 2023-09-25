import pickle
import cv2
import os
import base64
import numpy as np
from collections import deque
from flask import Flask, render_template, jsonify, request, url_for, redirect, send_file
import datetime

# from flask_cors import CORS


app = Flask(__name__)
# CORS(app)
# model = load_model('LRCN_model___Date_Time_2023_09_01__15_37_59___Loss_0.8564267158508301___Accuracy_0.49593496322631836.h5')  # Load your trained model
model = pickle.load(open('model.pkl', 'rb'))


IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
SEQUENCE_LENGTH = 20
# CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace", "Punch", "BreastStroke", ]
CLASSES_LIST = ['handclapping', 'handwaving', 'jogging', 'running', 'walking']
UPLOAD_FOLDER = 'static\\uploads' 
PREDICTED_FOLDER = 'static\\predicted'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTED_FOLDER'] = PREDICTED_FOLDER


def predict_on_video(video_path, output_file_path):
    '''
    This function will predict the class of frames in a sliding window fashion in the video passed to it.
    Args:
        video_path:         The path of the video on disk, whose frames are to be processed.
        output_file_path:   The path where the output video file with predicted class labels will be saved.
    '''
    video_reader = cv2.VideoCapture(video_path)
    
    # Get the Width and Height of the Video
    original_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), video_reader.get(cv2.CAP_PROP_FPS), (original_width, original_height))

    frame_queue = deque(maxlen=SEQUENCE_LENGTH)  # Sliding window queue

    while video_reader.isOpened():  

        # Read a frame from the video file.
        success, frame = video_reader.read()
        
        if not success:
            break
        
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame/255.0  # Normalize pixel values to [0, 1]

        frame_queue.append(normalized_frame)

        if len(frame_queue) == SEQUENCE_LENGTH:
            current_time2 = datetime.datetime.now()
            # If the window is filled, predict the class
            frames_to_predict = np.stack(frame_queue)  
            
            predicted_class_probabilities = model.predict(np.expand_dims(frames_to_predict, axis=0))[0]
            predicted_class = np.argmax(predicted_class_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_class]
            predicted_probability = predicted_class_probabilities[predicted_class]
            
            # Add the predicted class label and probability to the first frame in the window
            frame_with_label = frame_queue[0].copy()
            label_text = f"Class: {predicted_class_name}, Probability: {predicted_probability:.2f}"
            # print(label_text)
            cv2.putText(frame_with_label, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write the labeled frame to the output video
            video_writer.write(frame_with_label)

            # Remove the first frame from the window
            frame_queue.popleft()
            print(datetime.datetime.now() - current_time2)
        
        # After processing all frames, write any remaining frames in the queue
    for frame in frame_queue:
        video_writer.write(frame)

    video_writer.release()
    print("All Done")
    video_reader.release()


# model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        predicted_file_path = os.path.join(PREDICTED_FOLDER, 'predicted_' + os.path.basename(uploaded_file_path))
        
        file.save(uploaded_file_path)
        
        predict_on_video(uploaded_file_path, predicted_file_path)

        return send_file(predicted_file_path, mimetype='video/*')

if __name__ == '__main__':
    app.run(debug=True)
    