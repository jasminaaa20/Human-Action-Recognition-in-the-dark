import cv2
import face_recognition as fr
import pickle
import numpy as np
from multiprocessing import Pool
from datetime import datetime
import os
from collections import defaultdict


class FaceRecognition(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print('Creating the Face Recognition object')
            cls._instance = super(FaceRecognition, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        file = open('models/face_Encodings.p', 'rb')
        encodedListKnownWithNames = pickle.load(file)
        file.close()
        self.known_face_names, self.known_face_encodings = encodedListKnownWithNames
        self.output_main_folder = "static/faces/"
        self.unknown_counter = 0

    def set_output_folder(self, output_dir):
        self.output_folder = output_dir

    def identify(self, frame, height=0.5, width=0.5, is_Stream=True):
        identified_List = []
        img = cv2.resize(frame, (0, 0), None, height, width)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces_cur_frame = fr.face_locations(img)
        encoded_cur_frame = fr.face_encodings(img, faces_cur_frame)

        for encoded_faces, face_loc in zip(encoded_cur_frame, faces_cur_frame):
            matches = fr.compare_faces(
                self.known_face_encodings, encoded_faces)
            face_dis = fr.face_distance(
                self.known_face_encodings, encoded_faces)
            matchIndex = np.argmin(face_dis)

            if matches[matchIndex]:
                name = self.known_face_names[matchIndex]
            else:
                name = f'Unknown_{self.unknown_counter}'
                self.unknown_counter += 1
                self.known_face_encodings.append(encoded_faces)
                self.known_face_names.append(name)
            print(name)
            identified_List.append((name, face_loc))
            y2, x1, y1, x2 = face_loc
            y2, x1, y1, x2 = int(
                y2*(1/height)), int(x1*(1/width)), int(y1*(1/height)), int(x2*(1/width))
            face_image = frame[y2:y1, x2:x1]
            # face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            output_dir = os.path.join(self.output_folder, name)
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)

            if is_Stream:
                time_identified = datetime.now().strftime("%H%M%S")
            else:
                time_uploaded = output_dir.split('\\')[2].split('__')[-1]
                time_uploaded = time_uploaded.replace('_', ':')
                time_uploaded = datetime.strptime(time_uploaded, "%H:%M:%S")

                time_identified = datetime.now().strftime("%H%M%S") - time_uploaded

            file_name = os.path.join(
                output_dir, f'{face_dis[matchIndex]}_{time_identified}.jpeg')
            print(file_name, name)
            cv2.imwrite(file_name, face_image)

        return

    def getFaces(self):

        if self.output_folder is None:
            return None

        main_folder = self.output_folder
        if not os.path.exists(main_folder):
            os.makedirs(main_folder)
        identified = defaultdict(list)
        images_with_name = defaultdict(list)
        identified["base_folder"] = main_folder

        for name in os.listdir(main_folder):
            folder_path = os.path.join(main_folder, name)
            files = os.listdir(folder_path)
            sorted_files = sorted(files)
            first_5_files = sorted_files[:5]
            images_with_name[name] = first_5_files

        identified["images"] = images_with_name
        return identified
