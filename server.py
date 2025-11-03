import asyncio
import websockets
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import json
import time
import os

# Run YOLO on CPU
device = "cpu"
torch.cuda.is_available = lambda: False

# --- Model management ---
MODEL_DIR = r"weights"
current_model_name = None
model = None


def load_model(model_name):
    """Load YOLO model by name and move to device."""
    global model, current_model_name
    model_path = os.path.join(MODEL_DIR, model_name)
    model = YOLO(model_path)
    model.to(device)
    current_model_name = model_name
    print(f"✅ Loaded model: {model_name} on {device}")


def list_available_models():
    """Return a list of available .pt models in the weights folder."""
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
    return models


# Load the first available model on startup
available_models = list_available_models()
if not available_models:
    raise FileNotFoundError(f"No model weights found in '{MODEL_DIR}' folder.")
load_model(available_models[0])


# --- Utility functions ---
def decode_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def process_image(image_bytes):
    img = decode_image(image_bytes)
    start_time = time.time()
    results = model.predict(img, device=device)
    elapsed = round((time.time() - start_time) * 1000, 2)  # ms

    output = []
    for result in results:
        data = json.loads(result.to_json())
        output.extend(data)

    return json.dumps({
        "model": current_model_name,
        "time_ms": elapsed,
        "detections": output
    })


# --- WebSocket handler ---
async def handle_client(websocket):
    print("✅ Client connected")

    # Send available model list immediately
    model_list = list_available_models()
    await websocket.send(json.dumps({
        "type": "model_list",
        "available_models": model_list,
        "current_model": current_model_name
    }))
    print("📤 Sent model list to client")

    meta = None

    try:
        async for message in websocket:
            if isinstance(message, str):
                try:
                    data = json.loads(message)

                    # Handle model change request
                    if data.get("command") == "change_model":
                        requested = data.get("model_name")
                        if requested in list_available_models():
                            load_model(requested)
                            await websocket.send(json.dumps({
                                "type": "model_change_confirmation",
                                "message": f"Model changed to {requested}"
                            }))
                            print(f"🔄 Model switched to {requested}")
                        else:
                            await websocket.send(json.dumps({
                                "type": "error",
                                "message": f"Model '{requested}' not found."
                            }))
                        continue

                    # Otherwise, assume it's image metadata
                    meta = data
                    print(f"📝 Received header: {meta}")

                except json.JSONDecodeError:
                    print(f"Received text: {message}")
                    await websocket.send(f"Echo: {message}")

            elif isinstance(message, (bytes, bytearray)):
                if not meta:
                    print("⚠️ Received binary data without metadata")
                    continue

                msg_type = meta.get("type")
                if msg_type in ["file", "frame"]:
                    print(f"📸 Processing {msg_type} ({len(message)} bytes)")
                    response = await asyncio.to_thread(process_image, message)
                    await websocket.send(response)
                    print("✅ Sent detection result to client")
                else:
                    print(f"Unknown message type: {msg_type}")
                meta = None

    except websockets.ConnectionClosed:
        print("❌ Client disconnected")


# --- Run server ---
async def main():
    async with websockets.serve(handle_client, "localhost", 12345, max_size=50 * 1024 * 1024):
        print("🚀 Server running on ws://localhost:12345")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
