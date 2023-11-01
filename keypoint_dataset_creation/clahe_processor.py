import cv2
import numpy as np

class ClaheProcessor:
	def __init__(self, clip_limit=15, grid_size=(16, 16)):
    self.clip_limit = clip_limit
    self.grid_size = grid_size
    self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)

  def apply_clahe(self, image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    v = self.clahe.apply(v)
    hsv = np.dstack((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
