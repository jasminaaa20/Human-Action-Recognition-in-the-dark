from collections import deque
from datetime import datetime
import os
import pickle
import subprocess
import threading

import cv2
import numpy as np

from face_recognition_class import FaceRecognition
from clahe_processor import ClaheProcessor


class VideoProcessor:
    def __init__(self) -> None:
        self.action_recognition_model = pickle.load(
            open('models/model.pkl', 'rb'))
        self.image_height = 64
        self.image_width = 64
        self.sequence_length = 20
        self.classes_list = ['handclapping',
                             'handwaving', 'jogging', 'running', 'walking']
        self.upload_folder = 'static\\uploads'
        self.predicted_folder = 'static\\predicted'
        self.face_gallery = 'static\\faces'

        self.video_writer = None
        self.original_height = None
        self.original_width = None
        self.original_fps = None
        self.skip_frames = 6

        self.video_reader = None
        self.frame_queue = deque(maxlen=self.sequence_length)

        self.isStreaming = True

        self.text_thickness_proportion = 0.05  # Adjust as needed
        self.text_size_proportion = 0.02  # Adjust as needed
        self.predicted_class_name = "None"
        self.predicted_probability = 0.0

        self.face_recognition = None

        self.luminance = 50
        self.clahe_processor = ClaheProcessor()

    def get_video_processor(self, input_file):
        self.frame_queue.clear()
        self.isStreaming = True

        file_name = input_file.filename.split('.')[0]

        upload_file_path = os.path.join(self.upload_folder, file_name + '.mp4')
        self.output_file_path = os.path.join(
            self.predicted_folder, 'predicted_' + file_name + '.mp4')

        input_file.save(upload_file_path)
        self.video_reader = cv2.VideoCapture(upload_file_path)

        self.original_width = int(
            self.video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(
            self.video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fps = int(self.video_reader.get(cv2.CAP_PROP_FPS))
        self.skip_frames = int(self.original_fps*3 / 20)

        face_gallery = os.path.join(self.face_gallery, file_name)
        self.face_recognition = FaceRecognition()
        self.face_recognition.set_output_folder(face_gallery)

        print("video reader set")

        self.set_video_writer()
        return self.video_reader

    def set_streamer(self):
        self.frame_queue.clear()
        self.isStreaming = True
        self.video_reader = cv2.VideoCapture(0)
        self.output_file_path = os.path.join(self.predicted_folder, 'predicted_' + os.path.basename(datetime.now().strftime(
            "%Y_%m_%d__%H_%M_%S") + '.mp4'))

        face_gallery = os.path.join(
            self.face_gallery, datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.face_recognition = FaceRecognition()
        self.face_recognition.set_output_folder(face_gallery)

        self.set_video_writer()
        print("video streamer set")
        return self.video_reader

    def set_video_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_file_path, cv2.CAP_FFMPEG,
                                            fourcc, self.original_fps, (self.original_width, self.original_height), True)

    def get_skip_frames(self):
        return self.skip_frames

    def stop_streaming(self):
        self.isStreaming = False

    def process_frame(self, frame):

        self.update_luminance(frame)

        if self.luminance <= 50:
            frame = self.clahe_processor(frame)

        self.predicted_class_name = "None"
        self.predicted_probability = 0.0

        resized_frame = cv2.resize(
            frame, (self.image_height, self.image_width))
        normalized_frame = resized_frame / 255
        self.frame_queue.append(normalized_frame)

        if len(self.frame_queue) == self.sequence_length:
            frames_to_predict = np.stack(self.frame_queue)
            frames_to_predict = np.expand_dims(frames_to_predict, axis=0)

            predicted_class_probabilities = self.action_recognition_model.predict(
                frames_to_predict)[0]
            predicted_class_index = np.argmax(predicted_class_probabilities)

            if predicted_class_probabilities[predicted_class_index] > 0.5:
                self.predicted_class_name = self.classes_list[predicted_class_index]
                self.predicted_probability = predicted_class_probabilities[predicted_class_index]

            self.frame_queue.popleft()

    def write_frame(self, frame):
        if self.luminance <= 50:
            frame = self.clahe_processor(frame)

        font_size = (self.image_height * self.text_size_proportion)
        font_thickness = max(
            int(self.image_height * self.text_thickness_proportion), 1)
        label_text = f"Class: {self.predicted_class_name}, Probability: {self.predicted_probability:.4f}"
        cv2.putText(frame, label_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), font_thickness)
        self.video_writer.write(frame)

    def get_face_recognition_model(self):
        return self.face_recognition

    def release_all(self):
        success = True
        while success:
            success, _ = self.video_reader.read()

        self.video_reader.release()
        self.video_writer.release()
        cv2.destroyAllWindows()

    def convert_to_mp4(self):
        final_resting_place = os.path.join(self.output_file_path.split(
            '.')[0] + '~.mp4')

        command = [
            "E:\Programs\HandBrake\handbrakecli",  # Path to HandBrakeCLI
            "-i", self.output_file_path,  # Input file path
            "-o", final_resting_place,  # Output file path with .mp4 extension
            "-e", "x264",  # Use H.264 codec
        ]

        try:
            # Execute the HandBrakeCLI command
            subprocess.run(command)
        except Exception as e:
            print(
                f"Error compressing file: {self.output_file_path} ({str(e)})")

        os.remove(self.output_file_path)
        self.output_file_path = final_resting_place
        print("Converting Complete")

    def get_faces(self):
        return self.face_recognition.getFaces()

    def get_output_file_path(self):
        return self.output_file_path

    def update_luminance(self, frame):
        '''
        Given image, return a mean value of its colours. If its > 50 => light else dark
        '''
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        l = np.mean(gray_img)*100/255
        self.luminance = (self.luminance + l)/2
