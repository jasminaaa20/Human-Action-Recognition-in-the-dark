from datetime import datetime
import pickle
import cv2
import os
import numpy as np
from collections import deque
from flask import Flask, render_template, Response, jsonify, request, url_for, redirect, send_file
from flask_cors import CORS
import threading
from face_recognition_class import FaceRecognition
from multiprocessing import Process
import subprocess

app = Flask(__name__)
CORS(app) 
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

video_writer = None
video_reader = None
frame_queue = deque(maxlen=SEQUENCE_LENGTH)
isStreaming = False
n = 10

FR = FaceRecognition()

def streaming():
    print("start streaming")
    counter = 0
    output_dir_faces = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    while True:
        if not isStreaming:
            print("streaming end")
            success, frame = video_reader.read()
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n'
            break
        
        success, frame = video_reader.read()
        
        if not success:
            print("streaming end by Error")
            success, frame = video_reader.read()
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n'
            break
        
        
        if counter % 5 == 0:
            try:
                threading.Thread(target=FR.identify, args=(frame,output_dir_faces)).start()
            except:
                pass    
        
        predicted_class_name = "None"
        predicted_probability = 0.0
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame/255.0  # Normalize pixel values to [0, 1]

        frame_queue.append(normalized_frame)
        
        # If the window is filled, predict the class
        if len(frame_queue) == SEQUENCE_LENGTH:
            frames_to_predict = np.stack(frame_queue)  
            
            predicted_class_probabilities = model.predict(np.expand_dims(frames_to_predict, axis=0))[0]
            predicted_class = np.argmax(predicted_class_probabilities)
            
            if predicted_class_probabilities[predicted_class] > 0.5:
                predicted_class_name = CLASSES_LIST[predicted_class]
                predicted_probability = predicted_class_probabilities[predicted_class]
            
            frame_queue.popleft()
                
            # Add the predicted class label and probability to the first frame in the window
        label_text = f"Class: {predicted_class_name}, Probability: {predicted_probability:.4f}"
        font_size = 0.5
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)
        
        flag, encodedImage = cv2.imencode('.jpg', frame)
        
        if not flag:
            continue
        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        video_writer.write(frame)
        cv2.imshow("Video Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_reader.release()
    video_writer.release()
    cv2.destroyAllWindows()

def img_estim(img):
    '''
    Given image, return a mean value of its colours. If its > 50 => light else dark
    '''
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_img)*100/255

def process():
    print("start processing")
    counter = 0
    output_dir_faces = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    while True:
        
        success, frame = video_reader.read()
        
        if not success:
            print("processing end")
            break
        
        if counter % n == 0: # every n frames
            try:
                threading.Thread(target=FR.identify, args=(frame, output_dir_faces)).start()
            except:
                pass
            
            predicted_class_name = "None"
            predicted_probability = 0.0
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame/255.0
            frame_queue.append(normalized_frame)
            
            if len(frame_queue) == SEQUENCE_LENGTH:
                frames_to_predict = np.stack(frame_queue)
                predicted_class_probabilities = model.predict(np.expand_dims(frames_to_predict, axis=0))[0]
                predicted_class = np.argmax(predicted_class_probabilities)
                
                if predicted_class_probabilities[predicted_class] > 0.5:    
                    predicted_class_name = CLASSES_LIST[predicted_class]
                    predicted_probability = predicted_class_probabilities[predicted_class]
                
                frame_queue.popleft()
                
        label_text = f"Class: {predicted_class_name}, Probability: {predicted_probability:.4f}"
        font_size = 0.5
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)
        video_writer.write(frame)
    
    video_reader.release()
    video_writer.release()
    cv2.destroyAllWindows()
        
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
        uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename.split('.')[0]+'.mp4')
        output_file_path = os.path.join(PREDICTED_FOLDER, 'predicted_' + os.path.basename(uploaded_file_path))
        converted_file_path = output_file_path.split(".")[0] +"~.mp4"  
        
        file.save(uploaded_file_path)
        
        global video_reader
        global video_writer
        global frame_queue
        global n
        
        video_reader = cv2.VideoCapture(uploaded_file_path)
                
        # Get the Width and Height of the Video
        original_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = video_reader.get(cv2.CAP_PROP_FPS)
        n = (original_fps*6)/20
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
        video_writer = cv2.VideoWriter(output_file_path, cv2.CAP_FFMPEG,  fourcc, original_fps, (original_width, original_height), True)

        frame_queue = deque(maxlen=SEQUENCE_LENGTH)  # Sliding window queue
        
        process()
        # output_file_path = "F:\CS3501-Data_Science_and_Engineering_Project\Project Files\Human-Action-Recognition-in-the-dark\web_interface\API\static\predicted\predicted_50_FIRST_DATES_stand_u_cm_np1_fr_med_25.mp4"
        
        command = [
            "E:\Programs\HandBrake\handbrakecli",  # Path to HandBrakeCLI
            "-i", output_file_path,  # Input file path
            "-o", converted_file_path,  # Output file path with .mp4 extension
            "-e", "x264",  # Use H.264 codec
        ]
                
        try:
        # Execute the HandBrakeCLI command
            subprocess.run(command)
        except Exception as e:
            print(f"Error compressing file: {output_file_path} ({str(e)})")
        
        os.remove(output_file_path)
        return jsonify({"video_url": converted_file_path})
        # return jsonify({"video_url": "static\\uploads\\fetchme.avi"})

@app.route('/toggleStream', methods = ['POST'])
def predictWebCam():
    '''
    toggle the streaming on or off
    '''
    global isStreaming
    isStreaming = request.get_json() # toggle streaming or not 
    return "success"

@app.route('/stream', methods = ['GET'])
def stream():
    global isStreaming
    global video_reader
    global video_writer
    global frame_queue
    global n
    
    isStreaming = True
    video_reader = cv2.VideoCapture(0)
    
    original_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = video_reader.get(cv2.CAP_PROP_FPS)
    n = (original_fps*6)/20
    
    video_writer = cv2.VideoWriter(os.path.join(PREDICTED_FOLDER, 'predicted_' + os.path.basename(datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + '.mo4')), cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), original_fps, (original_width, original_height), True)
    
    frame_queue = deque(maxlen=SEQUENCE_LENGTH)

    print("start stream")
    return Response(streaming(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/getFaces', methods = ['GET'])
def getFaces():
    global FR
    faces = FR.getFaces()
    return jsonify(faces)

@app.route('/getImage/<path:file_path>', methods = ['GET'])
def getImage(file_path):
    return send_file(file_path, mimetype='image/jpeg')

@app.route('/getVideo/<path:file_path>', methods = ['GET'])
def getVideo(file_path):
    start = 0  # Start byte position
    length = os.path.getsize(file_path)  # Total length of the file
    end = length - 1  # End byte position
    headers = {
    'Content-Range': f'bytes {start}-{end}/{length}',
    'Accept-Ranges': 'bytes',
    'Content-Length': length,}
    return send_file(file_path, mimetype='video/mp4', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
    