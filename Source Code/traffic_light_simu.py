import time
from pyfirmata import Arduino, util

# Replace with your correct port (e.g., 'COM3' or '/dev/ttyACM0')
board = Arduino('COM8')

# Define the LED pins for each direction
traffic_lights = {
    "North": {"R": 2, "Y": 3, "G": 4},
    "East": {"R": 5, "Y": 6, "G": 7},
    "South": {"R": 8, "Y": 9, "G": 10},
    "West": {"R": 11, "Y": 12, "G": 13}
}

# Setup all pins as OUTPUT
for direction in traffic_lights.values():
    for pin in direction.values():
        board.digital[pin].mode = 1  # OUTPUT


def set_lights(active_direction):
    # Turn off all lights first
    for direction in traffic_lights.values():
        for pin in direction.values():
            board.digital[pin].write(0)

    # Set Red for all other directions
    for dir_name, pins in traffic_lights.items():
        if dir_name != active_direction:
            board.digital[pins["R"]].write(1)

    # Green ON for active direction
    active_pins = traffic_lights[active_direction]
    board.digital[active_pins["G"]].write(1)
    time.sleep(5)

    # Green OFF, Yellow ON
    board.digital[active_pins["G"]].write(0)
    board.digital[active_pins["Y"]].write(1)
    time.sleep(2)

    # Yellow OFF, Red ON
    board.digital[active_pins["Y"]].write(0)
    board.digital[active_pins["R"]].write(1)


try:
    while True:
        for direction in ["North", "East", "South", "West"]:
            set_lights(direction)

except KeyboardInterrupt:
    print("Exiting...")
    for direction in traffic_lights.values():
        for pin in direction.values():
            board.digital[pin].write(0)
    board.exit()
