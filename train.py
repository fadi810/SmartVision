import argparse
import torch
from ultralytics import YOLO


def benchmark(model, modelname, trained):
    """Benchmark model performance on GPU and CPU."""
    model.benchmark(
        data=args.data,
        batch=8,
        device=0,
        project="Benchmarks",
        name=f"{'trained' if trained else 'untrained'}_{modelname}_gpu"
    )
    model.benchmark(
        data=args.data,
        batch=8,
        device="cpu",
        project="Benchmarks",
        name=f"{'trained' if trained else 'untrained'}_{modelname}_cpu"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and benchmark YOLO models.")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML file")
    parser.add_argument("--model", type=str, required=True, help="Path to model file (.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarking before and after training")
    args = parser.parse_args()

    modelname = args.model.split("/")[-1].split(".")[0]
    model = YOLO(args.model)

    if args.benchmark:
        benchmark(model, modelname, False)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=4,
        device=0 if torch.cuda.is_available() else "cpu",
        half=True,
        augment=True,
        agnostic_nms=True
    )

    # Export trained model to OpenVINO and TorchScript
    model.export(format="openvino")
    model.export(format="torchscript")

    if args.benchmark:
        benchmark(model, modelname, True)
