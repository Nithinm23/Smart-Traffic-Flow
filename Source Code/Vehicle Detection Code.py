import numpy as np
import cv2
import math
from ultralytics import YOLO
import cvzone

# Initialize video capture for lane1
lane1 = cv2.VideoCapture("lane-1.mp4")

# Load YOLO model
model = YOLO("../../Yolo-Weights/yolov8n.pt")

# Define the classes
classNames = ["person", "bicycle", "car", "motorbike", "bus", "truck",
              "dog", "board", "cow", "cat", "sheep", "ambulance"]

# Load mask images (if needed)
mask1 = cv2.imread("lane-1.png")

# Define the lines for detection (not needed for vehicle detection alone)
start1 = [500, 250, 750, 250]
end1 = [150, 650, 1100, 650]

while True:
    # Read frames from the video
    success_1, road1 = lane1.read()

    if not success_1:
        break

    # Apply mask to focus on lane (optional)
    road1Region = cv2.bitwise_and(road1, mask1)

    # Perform YOLO detection
    results1 = model(road1Region, stream=True)

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

            # Filter to detect specific vehicle classes
            if currentclass1 in ["car", "motorbike", "truck", "bus", "bicycle", "ambulance"]:
                # Draw the bounding box and label
                cvzone.putTextRect(road1, f'{currentclass1} {conf}', (max(0, x1 + 20), max(0, y1 + 20)),
                                   scale=0.5, thickness=1, offset=3)
                cvzone.cornerRect(road1, (x1, y1, w, h), l=5)

    # Display the processed frame with detected vehicles
    cv2.imshow("Lane 1 - Vehicle Detection", road1)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
lane1.release()
cv2.destroyAllWindows()
