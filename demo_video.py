from ultralytics import YOLO

model = YOLO("yolo11s.pt")
results = model.predict(source='./inference/videos/test.mp4', stream=True, show=True, imgsz=640, save=True)

for result in results:
    print(result.boxes.xyxy)

