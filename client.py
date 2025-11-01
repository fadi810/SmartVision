import asyncio
import websockets
import time
import json

# async def send_messages():

#     start_time = time.time()  
    
#     uri = "ws://localhost:12345"

#     async with websockets.connect(uri) as websocket:
#         while True:

#             msg = await asyncio.to_thread(input, "Enter message (or 'exit' to quit): ")

#             if msg.lower() == "exit":
#                 break

#             await websocket.send(msg)
#             print(f"Sent: {msg}")
#             response = await websocket.recv()
#             elapsed = time.time() - start_time

#             print(f"Elapsed time: {elapsed:.3f} seconds")
#             print(f"Response: {response}")



async def send_image():

    start_time = time.time()  


    uri = "ws://localhost:12345"
    async with websockets.connect(uri) as websocket:


        with open(r"roboflow\valid\images\2007_000799_jpg.rf.ddf90d6a5d1625d6cca0fda1c6a91229.jpg", "rb") as img_file:
            data = img_file.read()


        await websocket.send(data)
        print("Image sent successfully!")

        response = await websocket.recv()
        response = json.loads(response)
        elapsed = time.time() - start_time

        print(f"Elapsed time: {elapsed:.3f} seconds")
        print([item["name"] for item in response])




# asyncio.run(send_messages())
asyncio.run(send_image())
