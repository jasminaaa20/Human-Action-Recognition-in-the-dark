from ultralytics import YOLO
import numpy as np

class KeypointDetector:
    def __init__(self, model_path, num_keypoints=17):
        self.model = YOLO(model_path)
        self.num_keypoints = num_keypoints

    def calculate_euclidean_distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def detect_keypoints(self, frames_list):
        prev_keypoints = None
        keypoints_list = []

        for frames in frames_list:
            keypoints_frame = None

            results = self.model(frames, stream=True)
            for result in results:
                if len(result) > 0:
                    keypoints_frame = result[0].keypoints.xyn.squeeze(0)

            if prev_keypoints is not None and keypoints_frame is not None:
                euclidean_distances = [
                    self.calculate_euclidean_distance(prev_point, current_point)
                    for prev_point, current_point in zip(prev_keypoints, keypoints_frame)
                ]
                keypoints_list.append(euclidean_distances)

            prev_keypoints = keypoints_frame if keypoints_frame is not None else prev_keypoints

        if keypoints_list:
            keypoints_array = np.array(keypoints_list)
        else:
            keypoints_array = np.zeros((self.SEQUENCE_LENGTH, self.num_keypoints))

        return keypoints_array
