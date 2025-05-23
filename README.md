### Driving Monitor in Python

Lightweight Python algorithm to automatically evaluate the fatigue level of any driver.

**Real time security system to implement on a vehicle**

Based on [Mediapipe's Facemesh](https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md), it tracks the driver's face and collects the most important face landmarks for the fatigue level estimation purpose. The system relies on the percentage of eye closure (PERCLOS), the mouth aspect ratio (MAR) and the head pose.

![Demonstration](demo/demo.gif)

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [How to use](#how-to-use)
- [How It Works](#how-it-works)
- [Libraries](#libraries)
- [To Do](#todo)

## Features

(This work is still not finished. Currently working on fine tuning the detection thresholds for multiple users)

- Real-time monitoring of driver's state
- Facial landmark detection for eye and mouth analysis
- Head pose estimation to assess head orientation
- Alerts for drowsy drivers
- Easy-to-use and customizable parameters for tuning the detection thresholds

## Requirements

1. Install dependencies:

```
pip3 install -r requirements.txt
```

## How to use

1. Clone this repository

```
git clone https://github.com/danielsousaoliveira/driving-monitor-python.git
```

2. Run the main script to start monitoring the driver's state:

```
python3 main.py
```

3. The system will use your default computer webcam to capture video and analyze facial features.

4. Press the 'Esc' key to exit the program.

## How It Works

The Driver Drowsiness Detection System uses the following components:

- **Head Posture**: This class estimates the driver's head pose using the MediaPipe FaceMesh module.

- **Face Detection**: This class evaluates the driver's facial features, such as eye openness, mouth state, and gaze, to detect signs of drowsiness.

- **Driver State Classifier**: This class combines the head pose and facial feature analysis to determine the driver's overall state, i.e., whether the driver is drowsy or alert.

## Libraries

- [opencv-python](https://github.com/opencv/opencv-python)
- [mediapipe](https://github.com/google/mediapipe)

## TODO

1. Fine tune detection thresholds
2. Increase robustness to deal with all mundane driving situations
3. Add miband implementation for alerts