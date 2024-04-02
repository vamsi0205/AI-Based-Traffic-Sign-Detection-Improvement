from ultralytics import YOLO
model = YOLO(r"V:\Hackathons\Techgium\Training\Training\best.pt")
results = model.train(
    data = r"V:\Hackathons\Techgium\Training\Training\data.yaml",
    imgsz = 640,
    epochs = 80,
    batch = 10,
    name = 'yolov8n_custom',
    augment=True
)



