import os
import numpy as np
from keypoint_detector import KeypointDetector
from frame_extractor import FrameExtractor

class DatasetCreator:
    def __init__(self, dataset_root, sequence_length, action_categories):
        self.dataset_root = dataset_root
        self.SEQUENCE_LENGTH = sequence_length
        self.action_categories = action_categories

    def get_action_label(self, action_folder):
        return self.action_categories.index(action_folder)

    def create_dataset(self):
        data_list = []
        label_list = []

        for action_folder in os.listdir(self.dataset_root):
            action_folder_path = os.path.join(self.dataset_root, action_folder)

            if os.path.isdir(action_folder_path):
                print(f"PROCESSING VIDEOS IN FOLDER: {action_folder}")

                for i, video_filename in enumerate(os.listdir(action_folder_path)):
                    if i % 50 == 0:
                        print(f"STILL PROCESSING VIDEOS IN FOLDER: {action_folder}")
                        print(f"PROCESSED VIDEOS SO FAR: {i}")

                    video_path = os.path.join(action_folder_path, video_filename)

                    video_data_extractor = FrameExtractor(self.SEQUENCE_LENGTH)
                    frames_list = video_data_extractor.frames_extraction(video_path)

                    keypoint_detector = KeypointDetector('yolov8n-pose.pt')
                    keypoints = keypoint_detector.detect_keypoints(frames_list)

                    label = self.get_action_label(action_folder)

                    data_list.append(keypoints)
                    label_list.append(label)

        data_array = np.array(data_list)
        label_array = np.array(label_list)

        return data_array, label_array
