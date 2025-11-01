import torch
from ultralytics import YOLO

def benchmark(model,modelname,trained):
    model.benchmark(data=r"roboflow/data.yaml", batch=8, device=0 , project=r"Benchmarks",name=f"{'trained' if trained == True else 'untrained'} {modelname} gpu")
    model.benchmark(data=r"roboflow/data.yaml", batch=8, device="cpu" , project=r"Benchmarks",name=f"{'trained' if trained == True else 'untrained'} {modelname} cpu")



if __name__ == "__main__":

    # model = YOLO(r"runs\detect\train\weights\best.pt")    #yolov8n
    model = YOLO(r"yolo11n.pt")

    benchmark(model,"yolo11n" , False)

    model.train(
        data="roboflow/data.yaml",  # Path to your dataset YAML file
        epochs=100,                 # Number of epochs
        imgsz=640,                  # Image size
        batch=16,                   # Batch size
        workers=4,                  # Number of dataloader workers
        device=0,                    # 0 for GPU, 'cpu' if no CUDA
        project=r"runs\detect\trainV11",
        half=True,
        augment=True,
        agnostic_nms = True
    )
    model.export(format="openvino")
    model.export(format="torchscript")
    
    benchmark(model,"yolov11n", True)
