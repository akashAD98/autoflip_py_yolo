import cv2
import torch

cap = cv2.VideoCapture('icegov.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0]

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
videoWriter = cv2.VideoWriter("xodance.mp4", fourcc, fps, size)

while True:
    ret, frame = cap.read()
    if not ret:
        break
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
            #print('width',dw)
            #print('height',dh)
            x = (int(pred[0]) +int(pred[1]))/2.0
            y = (int(pred[2]) + int(pred[3]))/2.0
            w = int(pred[1]) -int(pred[0])
            h = int(pred[3]) -int(pred[2])
            x = x*dw
            w = w*dw
            y = y*dh
            h = h*dh

            score = pred[4]
            #print("score",score)
            category_name = model.names[int(pred[5])]
            #print('category_name',category_name)
            category_id = pred[5]
            #print('category_id',category_id)
            
            Pesron_size = 200
            aspect_ratio = (x2 / y2)   # w //h 
        
            #print("aspect_ratio",aspect_ratio)
            
            #cv2.imshow(frame)
            # Calculate the width and height of the resized frame
            new_width = int(Pesron_size * aspect_ratio)
            new_height = int(Pesron_size / aspect_ratio)
            
            frame1 = cv2.resize(frame, (new_width, new_height))
            frame2 = cv2.resize(frame1, (300, 200))

            #print("frame",frame)

            cv2.imshow('frames',frame2)
            videoWriter.write(frame2)

    #frame.show()
    #results.show()
    #results.save()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
videoWriter.release()
cv2.destroyAllWindows()
