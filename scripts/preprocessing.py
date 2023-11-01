# Import the required libraries.
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import tensorflow_hub as hub

hub_model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = hub_model.signatures['serving_default']

mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils


def histogram_equalization(resized_frame):
    # Send a resized frame to this function, and it will return the histogram equalized frame.
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8, 8))
    img_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    alpha = 2.2  # 1.0 - 3.0
    beta = 50  # 0 - 100

    new_image = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=beta)
    gamma = 1.2
    look_up_table = np.empty((1, 256), np.uint8)
    for i in range(256):
        look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv2.LUT(new_image, look_up_table)
    final_img = clahe.apply(res)

    return final_img


def pose_estimation(resized_frame):
    # Send a resized frame to this function, and it will return the pose estimated frame.
    pose = mpPose.Pose(static_image_mode=False, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5, model_complexity=2)

    resized_frame.flags.writeable = False
    results = pose.process(resized_frame)
    resized_frame.flags.writeable = True

    return results


def show_media_pipe_pose(frame, pose_landmarks):
    # Original Frame, Landmarks from MediaPipe
    mpDraw.draw_landmarks(frame, pose_landmarks, mpPose.POSE_CONNECTIONS)
    lmList = []  # list of points for each frame
    for id, lm in enumerate(pose_landmarks.landmark):
        h, w, c = frame.shape
        cx, cy = int(lm.x*w), int(lm.y*h)
        lmList.append([id, cx, cy])

        # To highlight a specific landmark, uncomment the following lines.
        # if len(lmList) != 0:
        #     cv2.circle(resized_frame, (lmList[14][1], lmList[14][2]), 9, (255, 0, 0), cv2.FILLED)

    cv2.imshow("Image", frame)


def pose_estimaion6(frame):
    # Seperately Need to resize the frame to make prediction faster. Longerside should be > 256 and multiple of 32

    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 320, 512)
    input_img = tf.cast(img, dtype=tf.int32)

    # make prediction

    results = movenet(input_img)
    # each result has 56 points that is x,y,score * 17 + 5 bounding box coordinates
    keypoints = results["output_0"].numpy()[:, :, :51].reshape(
        6, 17, 3)  # 6 people, 17 key points, 3 for each key point

    # make sure it is confident enough to run through the model
    x, y, s = frame.shape
    input_keypoints = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    return input_keypoints
