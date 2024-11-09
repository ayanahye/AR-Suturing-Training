import os
import cv2
import numpy as np
from ultralytics import YOLO

# using a small segmentation model
model = YOLO('yolov5s.pt')
out_ = 'yolo-outputs2'
os.makedirs(out_, exist_ok=True)

# take image, run model, return detection results
def detect_tools(image):
    results = model(image)
    return results

# draw boxes on the image
def draw_boxes(image, results):
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        for box in boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return image

video_path = "Learn-to-Suture.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(os.path.join(out_, 'output.mp4'), 
                      cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # every 5 frames
    if frame_count % 5 == 0:
        results = detect_tools(frame)
        frame_with_boxes = draw_boxes(frame, results)
        
        # write frames
        out.write(frame_with_boxes)
        
        # save the frame as an image
        im_path = os.path.join(out_, f'frame_{frame_count}.jpg')
        cv2.imwrite(im_path, frame_with_boxes)
        
        print(f"processed frame {frame_count}")
        for result in results:
            for box in result.boxes:
                print(f"tool found at position: {box.xyxy[0].tolist()}, "
                      f"class: {model.names[int(box.cls.item())]}, confidence: {box.conf.item():.2f}")
    
    frame_count += 1

cap.release()
out.release()

# Results:
'''
Doesn't do very good, detects hands okay. Doesn't detect all the tools,
Most likely will need AR markers instead
'''
