import cv2
import torch
#from sort import Sort

from sort.tracker import SortTracker


# Load video and YOLOv5 model
cap = cv2.VideoCapture('dance_girls.mp4')
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

desired_aspect_ratio = 16/9

tracker_sortK = SortTracker()
print('tracker_sortK',tracker_sortK)

print("tracketr loadeda")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Run object detector on frame
    tracker_outputs = [None]
    results = model(frame, size=640)

    for image_id, prediction in enumerate(results.pred):
        tracker_type = tracker_sortK
        if tracker_type is not None:
            tracker_outputs[image_id] = tracker_sortK.update(prediction.cpu(),frame)

           # print('tracker_outputs[image_id]',tracker_outputs[image_id])

            print("tracker updaated.............>>>>>.")
            for output in tracker_outputs[image_id]:

                bbox, track_id, category_id, score = (
                            output[:4],
                            int(output[4]),
                            output[5],
                            output[6],
                        )
                category_name = model.names[int(category_id)]

                print('tracker id',track_id)
                print('category_name',category_name)
                print('score',score)

                # Show the reframed video
                x1, y1, x2, y2 = output[:4]  ##boxes[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.imshow("Reframed Video", frame)
            key = cv2.waitKey(1) & 0xFF
                
            # If the 'q' key is pressed, stop the loop
            if key == ord("q"):
                break
        
# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
