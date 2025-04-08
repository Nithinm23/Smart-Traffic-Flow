import numpy as np
import cv2
import math
from sort import *
from ultralytics import YOLO

# Initialize video capture for lane1
lane1 = cv2.VideoCapture("lane-1.mp4")

# Load YOLO model
model = YOLO("../../Yolo-Weights/yolov8n.pt")

# Define the classes
classNames = ["person", "bicycle", "car", "motorbike", "bus", "truck",
              "dog", "board", "cow", "cat", "sheep", "ambulance"]

# Load mask images
mask1 = cv2.imread("lane-1.png")

# Initialize trackers
tracker1 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define the lines for detection
start1 = [500, 250, 750, 250]
end1 = [150, 650, 1100, 650]

# Initialize vehicle counts
vehicleCount1 = []

while True:
    # Read frames from the video
    success_1, road1 = lane1.read()

    if not success_1:
        break

    # Apply mask to focus on lane
    road1Region = cv2.bitwise_and(road1, mask1)

    # Perform YOLO detection
    results1 = model(road1Region, stream=True)

    detection1 = np.empty((0, 5))

    # Process detections for lane1
    for r in results1:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currentclass1 = classNames[cls]

            if currentclass1 in ["car", "motorbike", "truck", "bus", "bicycle", "ambulance"]:
                currentArray1 = np.array([x1, y1, x2, y2, conf])
                detection1 = np.vstack((detection1, currentArray1))

    # Update tracker and track vehicles
    resultsTracker1 = tracker1.update(detection1)

    for result1 in resultsTracker1:
        x1, y1, x2, y2, Id1 = result1
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2  # Calculate center of bounding box

        # Vehicle enters the start line
        if start1[0] < cx < start1[2] and start1[1] - 15 < cy < start1[1] + 15:
            if Id1 not in vehicleCount1:
                vehicleCount1.append(Id1)

        # Vehicle crosses the end line
        if end1[0] < cx < end1[2] and end1[1] - 15 < cy < end1[1] + 15:
            if Id1 in vehicleCount1:
                vehicleCount1.remove(Id1)

    # Display the vehicle count on the frame
    print(f'Vehicle Count: {len(vehicleCount1)}')

    # Optional: If you want to see the frame in a window
    # cv2.imshow("Lane 1 - Vehicle Count", road1)
    
    # Exit condition (press 'q' to exit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
lane1.release()
cv2.destroyAllWindows()
