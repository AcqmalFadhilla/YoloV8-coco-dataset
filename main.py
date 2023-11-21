from ultralytics import YOLO
import cv2
import cvzone
import math

# video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640) #widht
cap.set(4, 640) #height

# Model YoloV8n with Dataset COCO
model = YOLO('Yolo-weight/yolov8n.pt')

className = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train',
             'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
             'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
             'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
             'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
             'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
             'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
             'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
             'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
             'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

while True:
    success, img = cap.read()
    result = model(img, stream=True)
    # bounding box
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 0, 0), 3)
            #confident
            conf = math.ceil((box.conf[0]*100))/100
            #class
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{className[cls]} {conf}', (max(35, x1), max(0, y1)))

    cv2.imshow("image", img)
    cv2.waitKey(1)
