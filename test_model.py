from ultralytics import YOLO

model = YOLO("runs/detect/train7/weights/best.pt")

results = model.predict(
    source="dataset/1.mp4",
    conf=0.25,
    save=True
)

print("Done")
