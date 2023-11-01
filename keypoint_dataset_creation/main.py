from clahe_processor import ClaheProcessor
from frame_extractor import FrameExtractor
from keypoint_detector import KeypointDetector
from dataset_creator import DatasetCreator
import numpy as np

# Define the path to your dataset root directory
dataset_root = '/content/drive/MyDrive/datasets/clips_v1.5'  # Replace with your dataset path

# Define the number of frames to extract from each video
sequence_length = 20

# Define the action categories
action_categories = [
    'Drink', 'Jump', 'Pick', 'Push', 'Run', 'Sit', 'Stand', 'Turn', 'Walk'
]

# Create an instance of the ClaheProcessor
clahe_processor = ClaheProcessor(clip_limit=15.0, tile_grid_size=(16, 16))

# Create an instance of the VideoDataExtractor
video_data_extractor = FrameExtractor(sequence_length, clahe_processor)

# Create an instance of the KeypointDetector
keypoint_detector = KeypointDetector('yolov8n-pose.pt')

# Create an instance of the DatasetCreator
dataset_creator = DatasetCreator(dataset_root, sequence_length, action_categories)

# Create the dataset
data_array, label_array = dataset_creator.create_dataset()

# Save the processed data and labels to file (optional)
np.save('processed_data.npy', data_array)
np.save('labels.npy', label_array)
