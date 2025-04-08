# Project: Smart Traffic Flow

## Overview
Smart Traffic Flow is a computer vision-based system designed to optimize traffic light signal timings. By calculating the number of vehicles at each traffic signal section using YOLOv8 for vehicle detection, the system adjusts signal timings to reduce long waiting times. Connected to an Arduino Uno microcontroller, the system operates the traffic lights (LEDs) from red to green based on real-time vehicle density, improving the overall traffic flow.

## Abstract
Smart Traffic Flow is a simple yet effective solution to address long waiting times at traffic signals. Installed cameras capture the real-time number of vehicles at each section of a traffic light, which traditionally uses fixed timing. By implementing YOLOv8-based vehicle detection and counting, the system dynamically adjusts the green light duration based on vehicle density, ensuring optimal traffic flow and reducing unnecessary delays. This project has the potential to be deployed in various traffic signal systems, significantly improving urban mobility.

## Table of Contents
- [Demo](#demo)
- [Components](#components)
- [Hardware](#hardware)
- [Code Base](#code-base)
- [Technologies Used](#technologies-used)
- [Result](#result)
- [Conclusion](#conclusion)

## Demo
[Demo Video](#Link-to-Demo)  
<p align="center"><b>Demo</b></p>

## Demo Photos
<p align="center">
  <img src="#Image-Link-1" width="200" />
  <img src="#Image-Link-2" width="200" />
  <img src="#Image-Link-3" width="200" />
  <img src="#Image-Link-4" width="200" />
</p>

## Components
Components Already Acquired/Owned:

| Component         | Quantity | Description |
| :---------------- | :------: | ----------- |
| Arduino Uno       | 1        | Microcontroller for controlling traffic signals |
| Camera Module     | 1        | Camera for real-time vehicle detection |
| LEDs (Red, Green) | 2        | LEDs for traffic signal indications |

## Hardware
<p align="center">
  <img src="#Pinout-Diagram-Link" width="300" />
</p>

## Code Base
- **Vehicle Detection Code**: YOLOv8-based detection for identifying vehicles in the traffic zone.
- **Vehicle Count Code**: Logic to count the number of vehicles in each section and calculate the required signal time.
- **Traffic Signal Control Code**: Logic to control the switching of traffic lights based on the vehicle count.

## Technologies Used
1. **OpenCV**: Open-source computer vision library used for real-time vehicle detection and analysis.
2. **YOLOv8**: Advanced object detection model for accurate and efficient vehicle identification.
3. **Arduino Uno**: The microcontroller used to operate the traffic light signals based on processed data.
4. **NumPy**: Python library for numerical operations to manage vehicle count data and timing calculations.

## Result
Smart Traffic Flow successfully demonstrated its ability to optimize traffic signal timings based on real-time vehicle density. Key results include:
- **Accurate Vehicle Detection**: The YOLOv8 model accurately detects vehicles in real-time, allowing for precise vehicle count estimation.
- **Dynamic Signal Timing**: The system adjusts the green light duration based on vehicle density, reducing unnecessary waiting times at traffic signals.
- **Scalability**: The system can be expanded to handle multiple traffic sections, with customizable vehicle detection zones and signal timing thresholds.

## Conclusion
Smart Traffic Flow provides an innovative solution to manage traffic congestion by dynamically adjusting signal timings. By leveraging advanced computer vision algorithms and integrating them with microcontroller-based traffic signal control, the system improves traffic flow and reduces wait times. This project has significant potential for urban traffic management, offering a scalable and efficient approach to optimize road safety and reduce traffic delays.
