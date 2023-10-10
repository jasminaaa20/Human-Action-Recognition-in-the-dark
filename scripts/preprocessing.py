# Import the required libraries.
import os
import cv2
import numpy as np

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
DATASET_DIR = "C:/Users/Benul Jayasekara/Desktop/ARID_v1_5_211015/clips_v1.5_avi"
CLASSES_LIST = ["Drink", "Jump"]#, "Pick", "Pour", "Push", "Run", "Sit", "Stand", "Turn", "Walk", "Wave"]


def frames_extraction(video_path):
    frames_list = []
    reader = cv2.VideoCapture(video_path)
    frames_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):

        reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = reader.read()
        if not success:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        alpha = 2.2  # 1.0 - 3.0
        beta = 50  # 0 - 100
        new_image = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=beta)
        gamma = 1.2
        look_up_table = np.empty((1, 256), np.uint8)
        for i in range(256):
            look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(new_image, look_up_table)
        clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8, 8))
        final_img = clahe.apply(res)
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)
    reader.release()
    return frames_list


def create_dataset():
    features = []
    labels = []
    video_files_paths = []

    for class_index, class_name in enumerate(CLASSES_LIST):

        print(f'Extracting Data of Class: {class_name}')
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))

        for file_name in files_list:

            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
            frames = frames_extraction(video_file_path)
            if len(frames) == SEQUENCE_LENGTH:
                # Append the data to their respective lists.
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

    # Converting the list to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels)

    return features, labels, video_files_paths

features, labels, video_files_paths = create_dataset()
