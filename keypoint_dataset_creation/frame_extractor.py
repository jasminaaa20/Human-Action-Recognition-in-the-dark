import cv2

class FrameExtractor:
    def __init__(self, sequence_length=20, clahe_processor=None):
        self.SEQUENCE_LENGTH = sequence_length
        self.clahe_processor = clahe_processor

    def frames_extraction(self, video_path):
        frames_list = []
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_window = max(int(frame_count / self.SEQUENCE_LENGTH), 1)

        for frame_counter in range(self.SEQUENCE_LENGTH):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_window)
            frame = video.read()[1]
            if frame is None:
                break
            if self.clahe_processor:
                frame = self.clahe_processor.apply_clahe(frame)
            frames_list.append(frame)
        video.release()
        return frames_list
