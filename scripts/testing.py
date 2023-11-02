# Python program to save a
# video using OpenCV


import cv2
import numpy as np

# Create an object to read
# from camera
video = cv2.VideoCapture("C:/Users/Benul Jayasekara/Desktop/ARID_v1_5_211015/clips_v1.5/Drink/Drink_24_26.mp4")

# We need to check if camera
# is opened previously or not
if (video.isOpened() == False):
    print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('output videos/Drink_24_26_color_enhancing.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size, 0)

while (True):
    ret, img = video.read()

    if ret == True:

        # Convert the image from BGR to HSV color space
        image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Adjust the hue, saturation, and value of the image
        # Adjusts the hue by multiplying it by 0.7
        image[:, :, 0] = image[:, :, 0] * 0.7
        # Adjusts the saturation by multiplying it by 1.5
        image[:, :, 1] = image[:, :, 1] * 1.5
        # Adjusts the value by multiplying it by 0.5
        image[:, :, 2] = image[:, :, 2] * 0.5

        # Convert the image back to BGR color space
        image2 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        imgGray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # filtered_img = cv2.medianBlur(imgGray, 7) # median blur

        # Remove noise using a Gaussian filter
        # filtered_img = cv2.GaussianBlur(imgGray, (7, 7), 0)

        # Create the sharpening kernel
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        # Sharpen the image
        # sharp1_img = cv2.filter2D(imgGray, -1, kernel) # very bad

        # Sharpen the image using the Laplacian operator
        # sharp2_img = cv2.Laplacian(imgGray, cv2.CV_64F) # WTF

        alpha = 1  # 1.0 - 3.0
        beta = 80  # 0 - 100
        new_image = cv2.convertScaleAbs(imgGray, alpha=alpha, beta=beta)
        gamma = 1.3
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(new_image, lookUpTable)
        clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8, 8))
        final_img = clahe.apply(res)

        # inverse
        # final_img = 255 - final_img  # bad

        # Write the frame into the
        # file 'filename.avi'
        result.write(final_img)

        # Display the frame
        # saved in the file
        cv2.imshow('Frame', final_img)

        # Press S on keyboard
        # to stop the process
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture and video
# write objects
video.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")