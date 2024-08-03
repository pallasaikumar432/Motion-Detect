import cv2  # OpenCV for image processing
import time  # For delay
import imutils  # For resizing images

# Initialize the webcam
cam = cv2.VideoCapture(0)  # 0 is the default camera ID
time.sleep(1)  # Allow the camera to warm up

# Initialize variables
firstFrame = None  # Variable to store the first frame
area = 500  # Minimum area for a contour to be considered motion

while True:
    # Read frame from camera
    _, img = cam.read()
    text = "Normal"  # Default text if no motion is detected
    img = imutils.resize(img, width=500)  # Resize frame to a width of 500 pixels

    # Convert to grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur for smoothing
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)

    # Capture the first frame
    if firstFrame is None:
        firstFrame = gaussianImg
        continue

    # Calculate the absolute difference between the first frame and current frame
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)
    # Threshold the difference image
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
    # Dilate the threshold image to fill in holes
    threshImg = cv2.dilate(threshImg, None, iterations=2)

    # Find contours
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Loop over the contours
    for c in cnts:
        # Ignore small contours
        if cv2.contourArea(c) < area:
            continue
        # Get bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # Draw rectangle around the contour
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object detected"  # Update text if motion is detected

    # Print the status
    print(text)
    # Display text on the frame
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # Show the frame
    cv2.imshow("cameraFeed", img)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF
    # Exit the loop if 'q' is pressed
    if key == ord("q"):
        break

# Release the webcam and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
