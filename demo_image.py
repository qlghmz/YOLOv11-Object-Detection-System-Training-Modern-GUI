from ultralytics import YOLO

model = YOLO("E:/code/yolov11/runs/detect/train6/weights/best.pt")
results = model.predict(source='E:/project/WaRP - Waste Recycling Plant Dataset/archive (11)/Warp-D/test/images/Monitoring_photo_2_test_25-Mar_11-44-17.jpg', stream=True, show=True, imgsz=640, save=True)

for result in results:
    print(result.boxes.xyxy)
