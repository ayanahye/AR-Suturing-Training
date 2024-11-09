import os
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# https://github.com/ultralytics/yolov5/releases

# using a small segmentation model
model = YOLO('yolov5s.pt')
out_ = 'yolo-outputs'
os.makedirs(out_, exist_ok=True)

# take image path, run model, return detection results
def detect_tools(image_path):
    results = model(image_path)
    return results

# draw boxes on the image
def draw_boxes(image, results):
    for result in results:
        # get the bounding coordinates of box in z1, y1, x2, y2
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        for box in boxes:
            # green rects on image thickness 2
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return image

dir_ = 'm2cai16-tool-locations/JPEGImages/'
image_files = sorted(
    [f for f in os.listdir(dir_) if f.lower().endswith('.jpg')],
    key=lambda x: os.path.getctime(os.path.join(dir_, x))
)

start_index = image_files.index('v09_044375_flip.jpg')
end_index = image_files.index('v09_047500.jpg')

selected_files = image_files[start_index:end_index+1:2]

image_paths = [os.path.join(dir_, f) for f in selected_files]
print(image_paths)

for i, image_path in enumerate(image_paths):
    results = model(image_path)
    
    image = cv2.imread(image_path)
    image_with_boxes = draw_boxes(image, results)
    
    im_path = os.path.join(out_, f'result_{i}.jpg')
    cv2.imwrite(im_path, image_with_boxes)

    print(f"detected objects in {image_path}:")
    for result in results:
        for box in result.boxes:
            print(f"tool found at position: {box.xyxy[0].tolist()}, "
      f"class: {model.names[int(box.cls.item())]}, confidence: {box.conf.item():.2f}")


