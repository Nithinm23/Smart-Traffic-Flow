import numpy as np
import time
import cv2
import cvzone
import math
from sort import *
from ultralytics import YOLO
from pyfirmata import Arduino, OUTPUT


# Initialize video capturec
lane1 = cv2.VideoCapture("lane-1.mp4")
lane2 = cv2.VideoCapture("lane-4.mp4")

# Load YOLO model
model = YOLO("../../Yolo-Weights/yolov8n.pt")

# Define the classes
classNames = ["person", "bicycle", "car", "motorbike", "bus", "truck",
              "dog", "board", "cow", "cat", "sheep", "ambulance"]

# Load mask images
mask1 = cv2.imread("lane-1.png")
mask2 = cv2.imread("lane-4.png")

# Define the combined video resolution
combined_width = 1280
combined_height = 720

# Initialize trackers
tracker1 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker2 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define the lines for detection
start1 = [500, 250, 750, 250]
end1 = [150, 650, 1100, 650]

start2 = [490, 250, 753, 250]
end2 = [150, 650, 1100, 650]

# Initialize vehicle counts
vehicleCount1 = []
vehicleCount2 = []

# Initialize the board for LED control
board = Arduino('COM9')  # Replace with your Arduino COM port

# Define pins for LEDs
RED_LED_PIN = 12
YELLOW_LED_PIN = 13
GREEN_LED_PIN = 11

WHITE_LED_PIN = 9
BLUE_LED_PIN = 10
PINK_LED_PIN = 8

# Constants for LED timing
LED_TIMING = {
    'RED_1': 60,
    'GREEN_1': 60,
    'YELLOW_BLINK': 1,
    'BLUE_1': 60,
    'WHITE_1': 60,
    'PINK': 50,

    'RED_2': 45,
    'GREEN_2': 45,
    'BLUE_2': 45,
    'WHITE_2': 45,


    'RED_3': 40,
    'GREEN_3': 40,
    'BLUE_3': 40,
    'WHITE_3': 40,
}

# Set up pins as outputs
board.digital[RED_LED_PIN].mode = OUTPUT
board.digital[YELLOW_LED_PIN].mode = OUTPUT
board.digital[GREEN_LED_PIN].mode = OUTPUT
board.digital[WHITE_LED_PIN].mode = OUTPUT
board.digital[BLUE_LED_PIN].mode = OUTPUT
board.digital[PINK_LED_PIN].mode = OUTPUT

# Variables for non-blocking timing
last_update_time = time.time()

# Timing configuration (in seconds)
def update_traffic_lights():
    global last_update_time

    current_time = time.time()
    elapsed_time = current_time - last_update_time

    # Determine LED timing based on vehicle counts
    if (len(vehicleCount1) == 0 and len(vehicleCount2) < 10) or (len(vehicleCount1) < 10 and len(vehicleCount2) == 0):
        timing_1 = 'RED_1'
        green_timing = 'GREEN_1'
        timing_2 = 'BLUE_1'
        green_timing_2 = 'WHITE_1'
        yellow_timing = 'YELLOW_BLINK'


    elif (10 <= len(vehicleCount1) <= 12 and len(vehicleCount2) <= 12) or (12 <= len(vehicleCount1) and len(vehicleCount2) <= 10):
        timing_1 = 'RED_2'
        green_timing = 'GREEN_2'
        timing_2 = 'BLUE_2'
        green_timing_2 = 'WHITE_2'
        yellow_timing = 'YELLOW_BLINK'

    elif (12 <= len(vehicleCount1) and len(vehicleCount2) <= 100) or (100 <= len(vehicleCount1) and len(vehicleCount2) <= 12):
        timing_1 = 'RED_3'
        green_timing = 'GREEN_3'
        timing_2 = 'BLUE_3'
        green_timing_2 = 'WHITE_3'
        yellow_timing = 'YELLOW_BLINK'

    else:
        timing_1 = 'RED_1'
        green_timing = 'GREEN_1'
        timing_2 = 'BLUE_1'
        green_timing_2 = 'WHITE_1'
        yellow_timing = 'YELLOW_BLINK'

    # Calculate total cycle duration and current time within the cycle
    cycle_duration = LED_TIMING[timing_1] + LED_TIMING[timing_2] + LED_TIMING[green_timing] + LED_TIMING[green_timing_2]
    time_in_cycle = elapsed_time % cycle_duration

    # Red and Blue LEDs in parallel
    if time_in_cycle < LED_TIMING[timing_1]:
        board.digital[RED_LED_PIN].write(1)
        board.digital[BLUE_LED_PIN].write(1)
        board.digital[GREEN_LED_PIN].write(0)
        board.digital[WHITE_LED_PIN].write(0)
        board.digital[PINK_LED_PIN].write(1)
        # Yellow LED blinking during RED phase
        blink_cycle = time_in_cycle % LED_TIMING[yellow_timing]
        if blink_cycle < LED_TIMING[yellow_timing] / 2:
            board.digital[YELLOW_LED_PIN].write(1)
        else:
            board.digital[YELLOW_LED_PIN].write(0)

    elif time_in_cycle < LED_TIMING[timing_1] + LED_TIMING[timing_2]:
        board.digital[RED_LED_PIN].write(0)
        board.digital[BLUE_LED_PIN].write(0)
        board.digital[GREEN_LED_PIN].write(1)
        board.digital[WHITE_LED_PIN].write(1)
        board.digital[PINK_LED_PIN].write(1)
        # Yellow LED off during GREEN phase
        board.digital[YELLOW_LED_PIN].write(0)

    elif time_in_cycle < LED_TIMING[timing_1] + LED_TIMING[timing_2] + LED_TIMING[green_timing]:
        board.digital[RED_LED_PIN].write(1)
        board.digital[BLUE_LED_PIN].write(1)
        board.digital[GREEN_LED_PIN].write(0)
        board.digital[WHITE_LED_PIN].write(0)
        board.digital[PINK_LED_PIN].write(1)
        # Yellow LED blinking during RED phase
        blink_cycle = time_in_cycle % LED_TIMING[yellow_timing]
        if blink_cycle < LED_TIMING[yellow_timing] / 2:
            board.digital[YELLOW_LED_PIN].write(1)
        else:
            board.digital[YELLOW_LED_PIN].write(0)

    elif time_in_cycle < LED_TIMING[timing_1] + LED_TIMING[timing_2] + LED_TIMING[green_timing] + LED_TIMING[green_timing_2]:
        board.digital[RED_LED_PIN].write(0)
        board.digital[BLUE_LED_PIN].write(0)
        board.digital[GREEN_LED_PIN].write(1)
        board.digital[WHITE_LED_PIN].write(1)
        board.digital[PINK_LED_PIN].write(1)
        # Yellow LED blinking during WHITE phase
        blink_cycle = time_in_cycle % LED_TIMING[yellow_timing]
        if blink_cycle < LED_TIMING[yellow_timing] / 2:
            board.digital[YELLOW_LED_PIN].write(1)
        else:
            board.digital[YELLOW_LED_PIN].write(0)

    else:
        board.digital[RED_LED_PIN].write(1)
        board.digital[BLUE_LED_PIN].write(1)
        board.digital[PINK_LED_PIN].write(1)
        board.digital[GREEN_LED_PIN].write(0)
        board.digital[WHITE_LED_PIN].write(0)
        board.digital[YELLOW_LED_PIN].write(0)

        # Update the last update time for the next cycle
    if elapsed_time - last_update_time > cycle_duration:
        last_update_time = current_time

while True:
    # Read frames from both videos
    success_1, road1 = lane1.read()
    success_2, road2 = lane2.read()

    if not success_1 or not success_2:
        break

    # Apply mask to focus on lane
    road1Region = cv2.bitwise_and(road1, mask1)
    road2Region = cv2.bitwise_and(road2, mask2)

    # Perform YOLO detection
    results1 = model(road1Region, stream=True)
    results2 = model(road2Region, stream=True)

    detection1 = np.empty((0, 5))
    detection2 = np.empty((0, 5))

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
                cvzone.putTextRect(road1, f'{currentclass1} {conf}', (max(0, x1 + 20), max(0, y1 + 20)),
                                   scale=0.5, thickness=1, offset=3)

                cvzone.cornerRect(road1, (x1, y1, w, h), l=5)
                currentArray1 = np.array([x1, y1, x2, y2, conf])
                detection1 = np.vstack((detection1, currentArray1))

    # Update tracker and draw lines for lane1
    resultsTracker1 = tracker1.update(detection1)
    cv2.line(road1, (start1[0], start1[1]), (start1[2], start1[3]), (0, 0, 255), 5)
    cv2.line(road1, (end1[0], end1[1]), (end1[2], end1[3]), (0, 255, 0), 5)

    for result1 in resultsTracker1:
        x1, y1, x2, y2, Id1 = result1
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(road1, (x1, y1, w, h), l=7, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(road1, f'{int(Id1)}', (max(0, x1), max(35, y1)),
                           scale=0.5, thickness=1, offset=3)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(road1, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

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

    # Update tracker and draw lines for lane2
    resultsTracker2 = tracker2.update(detection2)
    cv2.line(road2, (start2[0], start2[1]), (start2[2], start2[3]), (0, 0, 255), 5)
    cv2.line(road2, (end2[0], end2[1]), (end2[2], end2[3]), (0, 255, 0), 5)

    for result2 in resultsTracker2:
        x1, y1, x2, y2, Id2 = result2
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(road2, (x1, y1, w, h), l=7, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(road2, f'{int(Id2)}', (max(0, x1), max(35, y1)),
                           scale=0.5, thickness=1, offset=3)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(road2, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if start2[0] < cx < start2[2] and start2[1] - 15 < cy < start2[1] + 15:
            if Id2 not in vehicleCount2:
                vehicleCount2.append(Id2)
                cv2.line(road2, (start2[0], start2[1]), (start2[2], start2[3]), (0, 0, 255), 5)

        if end2[0] < cx < end2[2] and end2[1] - 15 < cy < end2[1] + 15:
            if Id2 in vehicleCount2:
                vehicleCount2.remove(Id2)
                cv2.line(road2, (end2[0], end2[1]), (end2[2], end2[3]), (0, 255, 0), 5)

    # Display vehicle counts
    cvzone.putTextRect(road1, f'Count: {len(vehicleCount1)}', (50, 50))
    cvzone.putTextRect(road2, f'Count: {len(vehicleCount2)}', (50, 50))

    # Update traffic lights based on vehicle counts
    update_traffic_lights()

    # Resize frames to match combined resolution
    height1, width1 = road1.shape[:2]
    height2, width2 = road2.shape[:2]

    new_width1 = combined_width // 2
    new_height1 = int(height1 * (new_width1 / width1))

    new_width2 = combined_width // 2
    new_height2 = int(height2 * (new_width2 / width2))

    road1 = cv2.resize(road1, (new_width1, new_height1))
    road2 = cv2.resize(road2, (new_width2, new_height2))

    if new_height1 < combined_height and new_height2 < combined_height:
        road1 = cv2.resize(road1, (new_width1, combined_height))
        road2 = cv2.resize(road2, (new_width2, combined_height))
    else:
        scaling_factor = combined_height / max(new_height1, new_height2)
        new_width1 = int(new_width1 * scaling_factor)
        new_width2 = int(new_width2 * scaling_factor)
        road1 = cv2.resize(road1, (new_width1, combined_height))
        road2 = cv2.resize(road2, (new_width2, combined_height))

    combined_frame = np.hstack((road2, road1))

    cv2.imshow('Combined Video', combined_frame)
    cv2.waitKey(1)

# Cleanup
lane1.release()
lane2.release()
cv2.destroyAllWindows()
board.exit()
