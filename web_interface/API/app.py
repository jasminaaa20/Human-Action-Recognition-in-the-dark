import pickle
import cv2
import os
import numpy as np
from collections import deque
from flask import Flask, render_template, jsonify, request, url_for, redirect, send_file
import datetime

app = Flask(__name__)
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
    cv2.namedWindow("Video with Prediction", cv2.WINDOW_NORMAL)
    
    # Get the Width and Height of the Video
    original_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = video_reader.get(cv2.CAP_PROP_FPS)

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), original_fps, (original_width, original_height), True)

    frame_queue = deque(maxlen=SEQUENCE_LENGTH)  # Sliding window queue

    while video_reader.isOpened():
        predicted_class_name = "None"
        predicted_probability = 0.0

        # Read a frame from the video file.
        success, frame = video_reader.read()
        
        if not success:
            break
        
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame/255.0  # Normalize pixel values to [0, 1]

        frame_queue.append(normalized_frame)
        predicted_class_name = "None"
        
        # If the window is filled, predict the class
        if len(frame_queue) == SEQUENCE_LENGTH:
            frames_to_predict = np.stack(frame_queue)  
            
            predicted_class_probabilities = model.predict(np.expand_dims(frames_to_predict, axis=0))[0]
            predicted_class = np.argmax(predicted_class_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_class]
            predicted_probability = predicted_class_probabilities[predicted_class]
            frame_queue.popleft()
                
            # Add the predicted class label and probability to the first frame in the window
        label_text = f"Class: {predicted_class_name}, Probability: {predicted_probability:.4f}"
        font_size = 0.5
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)
        
        # Write the labeled frame to the output video
        video_writer.write(frame)
        cv2.imshow('Video with Prediction', frame)
    
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
        # After processing all frames, write any remaining frames in the queue
    for frame in frame_queue:
        video_writer.write(frame)

    video_writer.release()
    video_reader.release()
    return True


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
        # sucess = predict_on_video(uploaded_file_path, predicted_file_path)

        return "sucess"
        return sucess
        # return send_file("static/predicted/predicted_Test.mp4", mimetype='video/*')

if __name__ == '__main__':
    app.run(debug=True)
    