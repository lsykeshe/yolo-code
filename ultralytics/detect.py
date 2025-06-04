from ultralytics import YOLO
 
 
model = YOLO('yolov8n.pt')

results = model('ultralytics/data/SCI.jpg', save=True)