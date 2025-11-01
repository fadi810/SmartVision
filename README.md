# SmartVision
This project provides a local object detection system using a Python WebSocket server for real-time inference and a simple frontend for uploading images, videos, or live camera feeds.
It also includes tools for model training and exploratory experiments.

## Overview

The project contains several components working together:
- **`server.py`** – Runs a WebSocket server using `websockets` and `ultralytics`. It receives images, performs YOLO inference (on CPU by default), and returns detection results in JSON format.
- **`index.html`** – Frontend interface for uploading images, streaming live video, and viewing detection results. The page displays bounding boxes and outputs model details, inference time, and object coordinates.
- **`client.py`** – A Python simulation of a WebSocket client, useful for testing the server without a browser.
- **`main.ipynb`** – Jupyter notebook used for exploratory data analysis, image augmentation, and noise injection during the model development phase.
- **`train.py`** – Handles YOLO model training. It accepts arguments for model configuration and dataset YAML file for flexible training setups.

server.py
Hosts a WebSocket server (websockets.serve) that receives image or video frames from clients.
It runs inference using a YOLO model (typically best_yolov8n.pt or best_yolov11n.pt) and sends back JSON results containing detections, bounding box coordinates, confidence scores, and model info.

index.html
A frontend interface for testing and debugging.
It includes three main buttons:

Upload Image: sends an image to the server for detection and displays results with boxes drawn.

Upload Video: sends a video file for processing.

Upload Active Video Feed: streams live camera frames (via Chrome or another browser) to the server for real-time inference.
The results and JSON logs are shown alongside the preview.

client.py
A simple simulated WebSocket client for testing without using the webpage.
It can send image data or video frames to the server and display responses in the console.

train.py
Script for training YOLO models.
It accepts command-line arguments for the dataset YAML file and model configuration.
Example:
