import torch
import cv2
from pathlib import Path
import datetime
import argparse
#from google.colab.patches import cv2_imshow

model_path = 'content/yolov5/yolov5s.pt'
img_path = 'content/yolov5/img/road.jpg'
output_path = 'content/result/yolo'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True)
    return parser

args = get_args().parse_args()
img_path = args.img

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Load input image
img = cv2.imread(img_path)

# Detect objects using YOLOv5
results = model(img, size=640)

# Get coordinates of bounding boxes
coordinates = []
counter = 1
for result in results.xyxy[0]:
    if result[5] == 2.0:  # Class 2 represents vehicles
        box = result[0:4].tolist()
        coordinates.append(box)

        # Label bounding box with number
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(counter), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        counter += 1

# Get the current time
current_time = datetime.datetime.now()

# Convert the time to a string in a desired format
time_string = current_time.strftime("%Y%m%d_%H%M%S")

save_file_basic = output_path + "/result_" + time_string
save_file_jpg = output_path + "/result_" + time_string + ".jpg"
save_file_txt = output_path + "/result_" + time_string + ".txt"

# Display image with bounding boxes
cv2.imwrite(save_file_jpg, img)

# Save coordinates as txt file
with open(save_file_txt, 'w') as f:
    for i, box in enumerate(coordinates):
        f.write(f"{' '.join([str(coord) for coord in box])}\n")
        
with open(output_path + "/result.txt", 'w', encoding='utf-8') as f:
    f.write(save_file_basic)

