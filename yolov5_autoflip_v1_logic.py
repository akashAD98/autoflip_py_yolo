import cv2
import yolov5
import torch
from utils import create_video_writer
cap = cv2.VideoCapture('../short_test.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
# Model
model = yolov5.load('yolov5s.pt')
model.classes = [0]

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)

video_writer = create_video_writer('../short_test.mp4', '../output', fps)
while True:
    frame_is_returned, frame = cap.read()
    if frame_is_returned:
        results = model(frame, size=640)
        for _, image_predictions_in_xyxy_format in enumerate(results.xywh):
            for pred in image_predictions_in_xyxy_format.cpu().detach().numpy():
                x1, y1, x2, y2 = (
                    int(pred[0]),
                    int(pred[1]),
                    int(pred[2]),
                    int(pred[3]),
                )
                bbox = [x1, y1, x2, y2]
                
                print("maximum_shape_person",max(bbox))
                # https://stackoverflow.com/questions/56115874/how-to-convert-bounding-box-x1-y1-x2-y2-to-yolo-style-x-y-w-h
                
                dw = 1./frame[0]
                dh = 1./frame[1]

                x = (int(pred[0]) + int(pred[1]))/2.0
                y = (int(pred[2]) + int(pred[3]))/2.0
                w = int(pred[1]) - int(pred[0])
                h = int(pred[3]) - int(pred[2])
                x = x*dw
                w = w*dw
                y = y*dh
                h = h*dh

                score = pred[4]
                category_name = model.names[int(pred[5])]
                category_id = pred[5]
                
                Pesron_size = 200
                aspect_ratio = (x2 / y2)   # w //h 

                # Calculate the width and height of the resized frame
                new_width = int(Pesron_size * aspect_ratio)
                new_height = int(Pesron_size / aspect_ratio)
                
                frame1 = cv2.resize(frame, (new_width, new_height))
                frame2 = cv2.resize(frame1, (300, 200))
        
            
    
        video_writer.write(frame1)
    
    else:
        break
   
video_writer.release()
cap.release()
