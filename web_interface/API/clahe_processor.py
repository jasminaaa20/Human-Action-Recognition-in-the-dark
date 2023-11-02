import cv2
import numpy as np


class ClaheProcessor:
    def __init__(self, clip_limit=15, grid_size=(16, 16)):
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.grid_size)

    def apply_clahe(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
        v = self.clahe.apply(v)
        hsv_img = np.dstack((h, s, v))
        frame = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return frame
