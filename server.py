import asyncio
import websockets
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import json
import time

# Run YOLO on CPU
device = "cpu"
torch.cuda.is_available = lambda: False

# Load YOLO model once
model = YOLO(r"best_yolov8n.pt")
model.to(device)
print(f"✅ Model loaded on: {next(model.model.parameters()).device}")

# --- Utility functions ---
def decode_image(image_bytes):
    """Decode binary image bytes into an OpenCV image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def process_image(image_bytes):
    """Run YOLO inference and return structured JSON results."""
    img = decode_image(image_bytes)
    start_time = time.time()
    results = model.predict(img, device=device)
    elapsed = round((time.time() - start_time) * 1000, 2)  # ms

    output = []
    for result in results:
        data = json.loads(result.to_json())
        output.extend(data)

    return json.dumps({
        "model": "YOLOv8n (CPU)",
        "time_ms": elapsed,
        "detections": output
    })

# --- WebSocket client handler ---
async def handle_client(websocket):
    print("✅ Client connected")
    buffer = None
    meta = None

    try:
        async for message in websocket:
            # Text message → metadata (JSON)
            if isinstance(message, str):
                try:
                    meta = json.loads(message)
                    print(f"📝 Received header: {meta}")
                except json.JSONDecodeError:
                    print(f"Received text: {message}")
                    await websocket.send(f"Echo: {message}")

            # Binary message → image or frame data
            elif isinstance(message, (bytes, bytearray)):
                if not meta:
                    print("⚠️ Received binary data without metadata")
                    continue

                msg_type = meta.get("type")
                if msg_type in ["file", "frame"]:
                    print(f"📸 Processing {msg_type} ({len(message)} bytes)")
                    response = await asyncio.to_thread(process_image, message)

                    # Send JSON string back to client
                    await websocket.send(response)
                    print("✅ Sent detection result to client")
                else:
                    print(f"Unknown message type: {msg_type}")
                meta = None  # Reset metadata after one image
    except websockets.ConnectionClosed:
        print("❌ Client disconnected")

# --- Run server ---
async def main():
    async with websockets.serve(handle_client, "localhost", 12345, max_size=50 * 1024 * 1024):
        print("🚀 Server running on ws://localhost:12345")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
