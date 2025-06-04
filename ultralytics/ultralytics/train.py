from ultralytics import YOLO

# Load a model
# yaml会自动下载
model = YOLO("ultralytics/cfg/models/v8/yolov8CARAFE.yaml")  # build a new model from scratch

 
# Train the model
results = model.train(data="coco128.yaml", epochs=100, batch = 2,imgsz=640,workers = 0,)


