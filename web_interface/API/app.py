from datetime import datetime
import pickle
import cv2
import os
from collections import deque
from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_cors import CORS
from pre_processing import VideoProcessor

app = Flask(__name__)
CORS(app)

Video_Processor = VideoProcessor()


def streaming():
    print("start Streaming")
    counter = 0
    video_reader = Video_Processor.set_streamer()
    skip_frames = Video_Processor.get_skip_frames()
    face_recognition = Video_Processor.get_face_recognition_model()

    while True:
        if not Video_Processor.isStreaming:
            print("streaming end")
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n'
            break

        success, frame = video_reader.read()
        if not success:
            print("streaming end by File")
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n'
            break

        if counter % skip_frames == 0:
            Video_Processor.process_frame(frame)

            if counter % ((skip_frames) ** 2) == 0:
                face_recognition.identify(frame)

        Video_Processor.write_frame(frame)
        flag, encodedImage = cv2.imencode('.jpg', frame)
        counter += 1
        if not flag:
            continue
        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    Video_Processor.release_all()
    Video_Processor.convert_to_mp4()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        video_reader = Video_Processor.get_video_processor(file)
        face_recognition = Video_Processor.get_face_recognition_model()

        skip_frames = Video_Processor.get_skip_frames()
        counter = 0

        while True:
            success, frame = video_reader.read()

            if not success:
                break

            if counter % skip_frames == 0:
                Video_Processor.process_frame(frame)
                Video_Processor.update_luminance(frame)

                if counter % ((skip_frames) ** 2) == 0:
                    face_recognition.identify(frame, is_Stream=False)

            Video_Processor.write_frame(frame)
            counter += 1

        Video_Processor.release_all()
        Video_Processor.convert_to_mp4()
        output_file_path = Video_Processor.get_output_file_path()
        return jsonify({"video_url": output_file_path})


@app.route('/toggleStream', methods=['POST'])
def predictWebCam():
    '''
    toggle the streaming on or off
    '''
    isStreaming = request.get_json()  # toggle streaming or not

    if not isStreaming:
        Video_Processor.stop_streaming()
        # Implement to send the processed video File
    else:
        print("Streaming")
    return "success"


@app.route('/stream', methods=['GET'])
def stream():
    return Response(streaming(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/getFaces', methods=['GET'])
def getFaces():
    faces = Video_Processor.get_faces()
    return jsonify(faces)


@app.route('/getImage/<path:file_path>', methods=['GET'])
def getImage(file_path):
    return send_file(file_path, mimetype='image/jpeg')


@app.route('/getVideo/<path:file_path>', methods=['GET'])
def getVideo(file_path):
    start = 0  # Start byte position
    length = os.path.getsize(file_path)  # Total length of the file
    end = length - 1  # End byte position
    headers = {
        'Content-Range': f'bytes {start}-{end}/{length}',
        'Accept-Ranges': 'bytes',
        'Content-Length': length, }
    return send_file(file_path, mimetype='video/mp4', as_attachment=False)


if __name__ == '__main__':
    app.run(debug=True)
