from ultralytics import YOLO

dataset_dir = "/home/river/workspace/experiment/fiap/FIAP-POS-TECH/fase05/roboflow-datasets-v6/data.yaml"  # Caminho do dataset
model = YOLO("yolo11n.pt")  # ou yolov8s.pt (mais preciso, porém mais pesado)

model.train(data=dataset_dir, epochs=30, imgsz=640)
