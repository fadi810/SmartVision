

# SmartVision
This project provides a local object detection system using a **Python WebSocket server** for real-time inference and a **simple frontend** for uploading images, videos, or live camera feeds.
It also includes tools for model training and exploratory experiments.

---

## Overview

* **`server.py`**
  Hosts a WebSocket server (`websockets.serve`) that receives image or video frames from clients.
  It runs inference using a YOLO model (typically `best_yolov8n.pt` or `best_yolov11n.pt`) and sends back JSON results containing detections, bounding box coordinates, confidence scores, and model info.

* **`index.html`**
  A frontend interface for testing and debugging.
  It includes three main buttons:

  * **Upload Image:** sends an image to the server for detection and displays results with boxes drawn.
  * **Upload Video:** sends a video file for processing.
  * **Upload Active Video Feed:** streams live camera frames (via Chrome or another browser) to the server for real-time inference.
    The results and JSON logs are shown alongside the preview.

* **`client.py`**
  A simple simulated WebSocket client for testing without using the webpage.
  It can send image data or video frames to the server and display responses in the console.

* **`train.py`**
  Script for training YOLO models.
  It accepts command-line arguments for the dataset YAML file, model path, and optional flags such as benchmarking.
  Example:

  ```bash
  python train.py --data path/to/data.yaml --model yolov11n.pt --epochs 100 --batch 16 --benchmark
  ```

**Arguments:**

* `--data`: Path to the dataset YAML file
* `--model`: Model file or name (e.g., `yolov8n.pt`, `yolov11n.pt`)
* `--epochs`: Number of training epochs (default: 100)
* `--batch`: Batch size (default: 16)
* `--device`: Device to use (`0` for GPU, `'cpu'` for CPU)
* `--benchmark`: Run model benchmarks before and after training (optional)

- **`main.ipynb`**
  Used for exploratory analysis, early-stage experiments, and image preprocessing.
  Includes steps for augmentation, noise addition, and visualizing dataset characteristics.

---

## How to Run

1. **Start the WebSocket Server**

   ```bash
   python server.py
   ```

   The server listens on `ws://localhost:12345`.

2. **Open the Frontend**
   Open `index.html` in your browser.
   You can upload an image, a video, or start your webcam feed for live detection.

3. **(Optional) Run the Client Script**

   ```bash
   python client.py
   ```

   Use this if you want to simulate sending data to the server programmatically instead of using the web interface.

4. **Train a Model**

   ```bash
   python train.py --data path/to/data.yaml --model yolov11n.pt --epochs 100 --batch 16 --benchmark
   ```

---

## Notes

* The `.pt` models used for inference are **`best_yolov8n.pt`** and **`best_yolov11n.pt`**.