import cv2
import torch
from sort import Sort


# Load video and YOLOv5 model
cap = cv2.VideoCapture('leo.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0]


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
videoWriter = cv2.VideoWriter("xodance.mp4", fourcc, fps, size)




print('model_loaded ...............')
# Desired aspect ratio

desired_aspect_ratio = 4/16



# Initialize object tracker
tracker = Sort()

print("tracketr loadeda")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Run object detector on frame
    results = model(frame, size=640)

    boxes = []
    confidences = []

    for _, image_predictions_in_xyxy_format in enumerate(results.xywh):
            for pred in image_predictions_in_xyxy_format.cpu().detach().numpy():
                x1, y1, x2, y2 = (
                    int(pred[0]),
                    int(pred[1]),
                    int(pred[2]),
                    int(pred[3]),
                )

                bbox = [x1, y1, x2, y2]
                score = pred[4]

                print("score",score)
                #print("score",score)
                category_name = model.names[int(pred[5])]
                #print('category_name',category_name)
                category_id = pred[5]
                
                #update bbox to tracker


                #####################################################  need help for this part not getting bbox 
                boxes = tracker.update(bbox)

                x, y, w, h = boxes[0]
                x_center = x + w/2
                y_center = y + h/2
                aspect_ratio = float(w) / h
    
                ###############################################

    # Check if object is still in center of frame
    if aspect_ratio < desired_aspect_ratio:
        # Calculate the new frame dimensions
        frame_height, frame_width, _ = frame.shape
        new_height = int(frame_width * desired_aspect_ratio)
        new_width = frame_width
        
        # Get the offset to center the object
        x_offset = int((frame_width - new_width) / 2)
        y_offset = int((frame_height - new_height) / 2)
        if abs(x_center - x_offset - new_width/2) > 70 or abs(y_center - y_offset - new_height/2) > 70:
            x_offset = int(x_center - new_width/2)
            y_offset = int(y_center - new_height/2)
        
        # Create a region of interest around the object
        roi = frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width]
        
        # Draw the bounding box around the object
        cv2.rectangle(roi, (x - x_offset, y - y_offset), (x + w - x_offset, y + h - y_offset), (0, 255, 0), 2)
    
    # Show the reframed video
    cv2.imshow("Reframed Video", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # If the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
        
# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
