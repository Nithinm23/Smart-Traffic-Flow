# Smart Traffic Flow

## Overview  
<p align="justify">
Smart Traffic Flow is a computer vision-based system designed to optimize traffic light signal timings at junctions. By calculating the number of vehicles at each junction using YOLOv8 for vehicle detection, the system adjusts signal timings to reduce long waiting times. Connected to an Arduino Uno microcontroller, the system operates the traffic lights (LEDs) from red to green based on real-time vehicle density, improving the overall traffic flow at junctions.

## Abstract  
<p align="justify">
Smart Traffic Flow is a simple yet effective solution to address long waiting times at junctions. Installed cameras capture the real-time number of vehicles at each section of a junction light, which traditionally uses fixed timing. By implementing YOLOv8-based vehicle detection and counting, the system dynamically adjusts the green light duration based on vehicle density, ensuring optimal traffic flow and reducing unnecessary delays. This project has the potential to be deployed in various junction traffic signal systems, significantly improving urban mobility.

## Table of Contents  
- [Demo](#demo)  
- [Components](#components)  
- [Hardware](#hardware)  
- [Code Base](#code-base)  
- [Technologies Used](#technologies-used)  
- [Result](#result)  
- [Conclusion](#conclusion)  

## Demo  

## Demo Photos  
<p align="center">  
  <img src="https://github.com/user-attachments/assets/6d5d369e-6ef9-4854-9c97-7aaae29b169f" width="200" />  
  <img src="https://github.com/user-attachments/assets/ada3d6d2-94dc-4108-8703-77452f81fad3" width="200" /> 
  <img width="1909" height="1035" alt="Screenshot 2024-08-26 155630" src="https://github.com/user-attachments/assets/ce3184c1-5490-4703-9955-4e21544e59af" />
  
</p>

# Libraries

| Libraries | Description |
| :---         | :---      |
| OpenCV | Handles video capture, image processing, and visualization |
| NumPy | Used for numerical operations and array manipulation |
| cvzone | Used for simplifying computer vision tasks and overlays |
| TensorFlow/Keras | Deep learning framework used for model development |



## Components  
Components Already Acquired/Owned:  

| Component         | Quantity | Description |  
| :---------------- | :------: | ----------- |  
| Arduino Uno       | 1        | Microcontroller for controlling traffic signals at junctions |  
| Camera Module     | 1        | Camera for real-time vehicle detection at the junction |  
| LEDs (Red, Green) | 2        | LEDs for traffic signal indications at the junction |  


## Code Base  
- **Vehicle Detection Code**: YOLOv8-based detection for identifying vehicles in the junction area.  
- **Vehicle Count Code**: Logic to count the number of vehicles in each section of the junction and calculate the required signal time.  
- **Traffic Signal Control Code**: Logic to control the switching of traffic lights at the junction based on the vehicle count.  

## Technologies Used  
1. **OpenCV**: Open-source computer vision library used for real-time vehicle detection and analysis at junctions.  
2. **YOLOv8**: Advanced object detection model for accurate and efficient vehicle identification at junctions.  
3. **Arduino Uno**: The microcontroller used to operate the traffic light signals at the junction based on processed data.  
4. **NumPy**: Python library for numerical operations to manage vehicle count data and timing calculations at the junction.  

## Result  
Smart Traffic Flow successfully demonstrated its ability to optimize traffic signal timings based on real-time vehicle density at junctions. Key results include:  
- **Accurate Vehicle Detection**: The YOLOv8 model accurately detects vehicles in real-time, allowing for precise vehicle count estimation at junctions.  
- **Dynamic Signal Timing**: The system adjusts the green light duration based on vehicle density at the junction, reducing unnecessary waiting times.  
- **Scalability**: The system can be expanded to handle multiple junctions, with customizable vehicle detection zones and signal timing thresholds.  

## Conclusion  
<p align="justify">
Smart Traffic Flow provides an innovative solution to manage traffic congestion at junctions by dynamically adjusting signal timings. By leveraging advanced computer vision algorithms and integrating them with microcontroller-based traffic signal control, the system improves traffic flow and reduces wait times. This project has significant potential for urban traffic management, offering a scalable and efficient approach to optimize road safety and reduce traffic delays at junctions.
