import numpy as np
import math
import cv2
import cvzone
from sort import Sort
from ultralytics import YOLO
from pyfirmata import Arduino, util
import time

board = Arduino('COM8')
for pin in range(2, 14):
    board.digital[pin].mode = 1

# Initialize video capture for each lane
lane1 = cv2.VideoCapture("lane-1.mp4")
lane2 = cv2.VideoCapture("lane-2.mp4")
lane3 = cv2.VideoCapture("lane-3.mp4")
lane4 = cv2.VideoCapture("lane-4.mp4")

# Load YOLO model
model = YOLO("yolov8n.pt")

# Class names for YOLO detections
classNames = ["person", "bicycle", "car", "motorbike", "bus", "truck", "dog", "board", "cow", "cat", "sheep", "ambulance"]

# Masks for lanes
mask1 = cv2.imread("lane-1.png")
mask2 = cv2.imread("lane-2.png")
mask3 = cv2.imread("lane-3.png")
mask4 = cv2.imread("lane-4.png")

# Define combined video resolution
combined_width = 1080
combined_height = 520

# Initialize trackers for each lane
tracker1 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker2 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker3 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker4 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define start and end lines for each lane
start1 = [500, 250, 750, 250]
end1 = [150, 650, 1100, 650]

start2 = [490, 250, 753, 250]
end2 = [150, 650, 1100, 650]

start3 = [490, 250, 753, 250]
end3 = [150, 650, 1100, 650]

start4 = [490, 250, 753, 250]
end4 = [150, 650, 1100, 650]

# Initialize vehicle counts for each lane
vehicleCount1 = []
vehicleCount2 = []
vehicleCount3 = []
vehicleCount4 = []

# Example dictionary to hold all counts
lane_counts = {
    "North": len(vehicleCount1),
    "East": len(vehicleCount2),
    "South": len(vehicleCount3),
    "West": len(vehicleCount4)
}

# Sort lanes based on vehicle count
sorted_lanes = sorted(lane_counts, key=lane_counts.get, reverse=True)
active_lane = sorted_lanes[0]  # Lane with the highest count
# Function to light traffic lights based on active lane
# Turn off all LEDs
for direction_pins in pin_map.values():
    for pin in direction_pins.values():
        board.digital[pin].write(0)

# Set RED to others
for direction, pins in pin_map.items():
    if direction != active_lane:
        board.digital[pins["R"]].write(1)


# Set GREEN for active lane
board.digital[pin_map[active_lane]["G"]].write(1)
time.sleep(5)
board.digital[pin_map[active_lane]["G"]].write(0)

# Set YELLOW before turning RED
board.digital[pin_map[active_lane]["Y"]].write(1)
time.sleep(2)
board.digital[pin_map[active_lane]["Y"]].write(0)
board.digital[pin_map[active_lane]["R"]].write(1)

def control_traffic_lights(active_lane):
    pin_map = {
        "North": {"R": 2, "Y": 3, "G": 4},
        "East": {"R": 5, "Y": 6, "G": 7},
        "South": {"R": 8, "Y": 9, "G": 10},
        "West": {"R": 11, "Y": 12, "G": 13}
    }

while True:
    # Read frames from each lane
    success_1, road1 = lane1.read()
    success_2, road2 = lane2.read()
    success_3, road3 = lane3.read()
    success_4, road4 = lane4.read()

    if not success_1 or not success_2 or not success_3 or not success_4:
        break

    # Apply masks to focus on specific lane areas
    road1Region = cv2.bitwise_and(road1, mask1)
    road2Region = cv2.bitwise_and(road2, mask2)
    road3Region = cv2.bitwise_and(road3, mask3)
    road4Region = cv2.bitwise_and(road4, mask4)

    # Perform YOLO object detection on each lane
    results1 = model(road1Region, stream=True)
    results2 = model(road2Region, stream=True)
    results3 = model(road3Region, stream=True)
    results4 = model(road4Region, stream=True)

    detection1 = np.empty((0, 5))
    detection2 = np.empty((0, 5))
    detection3 = np.empty((0, 5))
    detection4 = np.empty((0, 5))

    # Process detections for lane1
    for t in results1:
        boxes = t.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currentclass1 = classNames[cls]

            if currentclass1 in ["car", "motorbike", "truck", "bus", "bicycle", "ambulance"]:
                cvzone.putTextRect(road1, f'{currentclass1} {conf}', (max(0, x1 + 20), max(0, y1 + 20)),
                                   scale=0.5, thickness=1, offset=3)
                cvzone.cornerRect(road1, (x1, y1, w, h), l=5)
                currentArray1 = np.array([x1, y1, x2, y2, conf])
                detection1 = np.vstack((detection1, currentArray1))

    resultsTracker1 = tracker1.update(detection1)
    cv2.line(road1, (start1[0], start1[1]), (start1[2], start1[3]), (0, 0, 255), 5)
    cv2.line(road1, (end1[0], end1[1]), (end1[2], end1[3]), (0, 255, 0), 5)

    for result1 in resultsTracker1:
        x1, y1, x2, y2, Id1 = result1
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        if start1[0] < cx < start1[2] and start1[1] - 15 < cy < start1[1] + 15:
            if Id1 not in vehicleCount1:
                vehicleCount1.append(Id1)
                cv2.line(road1, (start1[0], start1[1]), (start1[2], start1[3]), (0, 0, 255), 5)

        if end1[0] < cx < end1[2] and end1[1] - 15 < cy < end1[1] + 15:
            if Id1 in vehicleCount1:
                vehicleCount1.remove(Id1)
                cv2.line(road1, (end1[0], end1[1]), (end1[2], end1[3]), (0, 255, 0), 5)

    # Process detections for lane2
    for t in results2:
        boxes = t.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currentclass2 = classNames[cls]

            if currentclass2 in ["car", "motorbike", "truck", "bus", "bicycle", "ambulance"]:
                cvzone.putTextRect(road2, f'{currentclass2} {conf}', (max(0, x1 + 20), max(0, y1 + 20)),
                                   scale=0.5, thickness=1, offset=3)
                cvzone.cornerRect(road2, (x1, y1, w, h), l=5)
                currentArray2 = np.array([x1, y1, x2, y2, conf])
                detection2 = np.vstack((detection2, currentArray2))

    resultsTracker2 = tracker2.update(detection2)
    cv2.line(road2, (start2[0], start2[1]), (start2[2], start2[3]), (0, 0, 255), 5)
    cv2.line(road2, (end2[0], end2[1]), (end2[2], end2[3]), (0, 255, 0), 5)

    for result2 in resultsTracker2:
        x1, y1, x2, y2, Id2 = result2
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        if start2[0] < cx < start2[2] and start2[1] - 15 < cy < start2[1] + 15:
            if Id2 not in vehicleCount2:
                vehicleCount2.append(Id2)
                cv2.line(road2, (start2[0], start2[1]), (start2[2], start2[3]), (0, 0, 255), 5)

        if end2[0] < cx < end2[2] and end2[1] - 15 < cy < end2[1] + 15:
            if Id2 in vehicleCount2:
                vehicleCount2.remove(Id2)
                cv2.line(road2, (end2[0], end2[1]), (end2[2], end2[3]), (0, 255, 0), 5)

    # Process detections for lane3
    for t in results3:
        boxes = t.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currentclass3 = classNames[cls]

            if currentclass3 in ["car", "motorbike", "truck", "bus", "bicycle", "ambulance"]:
                cvzone.putTextRect(road3, f'{currentclass3} {conf}', (max(0, x1 + 20), max(0, y1 + 20)),
                                   scale=0.5, thickness=1, offset=3)
                cvzone.cornerRect(road3, (x1, y1, w, h), l=5)
                currentArray3 = np.array([x1, y1, x2, y2, conf])
                detection3 = np.vstack((detection3, currentArray3))

    resultsTracker3 = tracker3.update(detection3)
    cv2.line(road3, (start3[0], start3[1]), (start3[2], start3[3]), (0, 0, 255), 5)
    cv2.line(road3, (end3[0], end3[1]), (end3[2], end3[3]), (0, 255, 0), 5)

    for result3 in resultsTracker3:
        x1, y1, x2, y2, Id3 = result3
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        if start3[0] < cx < start3[2] and start3[1] - 15 < cy < start3[1] + 15:
            if Id3 not in vehicleCount3:
                vehicleCount3.append(Id3)
                cv2.line(road3, (start3[0], start3[1]), (start3[2], start3[3]), (0, 0, 255), 5)

        if end3[0] < cx < end3[2] and end3[1] - 15 < cy < end3[1] + 15:
            if Id3 in vehicleCount3:
                vehicleCount3.remove(Id3)
                cv2.line(road3, (end3[0], end3[1]), (end3[2], end3[3]), (0, 255, 0), 5)

    # Process detections for lane4
    for t in results4:
        boxes = t.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currentclass4 = classNames[cls]

            if currentclass4 in ["car", "motorbike", "truck", "bus", "bicycle", "ambulance"]:
                cvzone.putTextRect(road4, f'{currentclass4} {conf}', (max(0, x1 + 20), max(0, y1 + 20)),
                                   scale=0.5, thickness=1, offset=3)
                cvzone.cornerRect(road4, (x1, y1, w, h), l=5)
                currentArray4 = np.array([x1, y1, x2, y2, conf])
                detection4 = np.vstack((detection4, currentArray4))

    resultsTracker4 = tracker4.update(detection4)
    cv2.line(road4, (start4[0], start4[1]), (start4[2], start4[3]), (0, 0, 255), 5)
    cv2.line(road4, (end4[0], end4[1]), (end4[2], end4[3]), (0, 255, 0), 5)

    for result4 in resultsTracker4:
        x1, y1, x2, y2, Id4 = result4
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        if start4[0] < cx < start4[2] and start4[1] - 15 < cy < start4[1] + 15:
            if Id4 not in vehicleCount4:
                vehicleCount4.append(Id4)
                cv2.line(road4, (start4[0], start4[1]), (start4[2], start4[3]), (0, 0, 255), 5)

        if end4[0] < cx < end4[2] and end4[1] - 15 < cy < end4[1] + 15:
            if Id4 in vehicleCount4:
                vehicleCount4.remove(Id4)
                cv2.line(road4, (end4[0], end4[1]), (end4[2], end4[3]), (0, 255, 0), 5)

    # Display vehicle counts for all lanes
    cvzone.putTextRect(road1, f'Count: {len(vehicleCount1)}', (50, 50), scale=5, thickness=5)
    cvzone.putTextRect(road2, f'Count: {len(vehicleCount2)}', (50, 50), scale=5, thickness=5)
    cvzone.putTextRect(road3, f'Count: {len(vehicleCount3)}', (50, 50), scale=5, thickness=5)
    cvzone.putTextRect(road4, f'Count: {len(vehicleCount4)}', (50, 50), scale=5, thickness=5)

    # Resize and combine frames for display
    height1, width1 = road1.shape[:2]
    height2, width2 = road2.shape[:2]
    height3, width3 = road3.shape[:2]
    height4, width4 = road4.shape[:2]

    road1_resized = cv2.resize(road1, (combined_width // 2, combined_height // 2))
    road2_resized = cv2.resize(road2, (combined_width // 2, combined_height // 2))
    road3_resized = cv2.resize(road3, (combined_width // 2, combined_height // 2))
    road4_resized = cv2.resize(road4, (combined_width // 2, combined_height // 2))

    # Stack frames vertically
    top_row = np.hstack((road1_resized, road2_resized))
    bottom_row = np.hstack((road3_resized, road4_resized))

    final_output = np.vstack((top_row, bottom_row))

    # Show the final output
    cv2.imshow("Vehicle Counting", final_output)

    control_traffic_lights(active_lane)

    # Break if the user presses the 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break



# Release video captures and close windows
lane1.release()
lane2.release()
lane3.release()
lane4.release()
cv2.destroyAllWindows()
