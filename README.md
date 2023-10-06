# Action Recognition in the Dark using ARID Dataset

The Action Recognition in the Dark (ARID) dataset focuses on human action recognition in challenging lighting conditions, such as low-light and darkness. This project employs ARID v1.5 to train a model to detect human actions under these lighting conditions.

## Dataset Overview
The ARID dataset consists of video clips featuring 11 distinct human actions performed in various scenes, both indoor and outdoor, under varying lighting conditions. More details about the ARID v1.5 dataset can be found above.

## Preprocessing Steps

### 1. Frame Extraction from Videos

From each video, 20 frames are extracted to form a list of frames per video, serving as the primary dataset elements.

![Frame Extraction](path/to/frame_extraction_image.jpg)

### 3. Histogram Equalization

To improve visibility, reduce noise, and standardize frames, histogram equalization is applied to each frame.

![Histogram Equalization](path/to/histogram_equalization_image.jpg)

### 4. Key-point Detection using YOLOv8-pose

Key-point information is extracted from each frame using the YOLOv8-pose key-point detection model.

![Key-point Detection](path/to/keypoint_detection_image.jpg)

## Model Architecture

The model architecture utilizes a Long-term Recurrent Convolutional Networks (LRCN) approach. The detailed code for the architecture can be found above.

![Model Architecture](path/to/model_architecture_image.jpg)

## Model Compilation and Training

The model is compiled using categorical cross-entropy loss, optimized with Adam optimizer. The training history is saved for plotting and evaluation purposes.

## Model Evaluation

The model is evaluated using the test dataset, and a confusion matrix is generated to analyze its performance:

|       | Walking | Pushing | Turning | ... | Picking |
|-------|---------|---------|---------|-----|---------|
| Walking | xx  | xx      | xx      | ... | xx      |
| Pushing | xx  | xx      | xx      | ... | xx      |
| ...     | ...  | ...     | ...     | ... | ...     |
| Picking | xx  | xx      | xx      | ... | xx      |

_Note: Replace the "xx" with actual confusion matrix values._

## Results

The loss and accuracy curves of the model during training can be seen below:

### Loss Curve

![Loss Curve](path/to/loss_curve_image.jpg)

### Accuracy Curve

![Accuracy Curve](path/to/accuracy_curve_image.jpg)

---

## Conclusion

This project has successfully built a model to detect and classify human actions in challenging lighting conditions using the ARID v1.5 dataset. The LRCN approach combined with strategic preprocessing steps has allowed for improved accuracy and performance.
