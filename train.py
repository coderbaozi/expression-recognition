from ultralytics import YOLO
model = YOLO('./pre-train-model/yolov8s.pt')

def setup():
  result = model.train(data='./datasets/expression.yaml',epochs=100, imgsz=640)
  return result

if __name__ == '__main__':
  setup()