import cv2
import torch

cap = cv2.VideoCapture('../short_test.mp4')
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0]

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)

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
            score = pred[4]
            category_name = model.names[int(pred[5])]
            category_id = pred[5]
    
    #crops_person = results.crop(save=True)

    #face = max(results.crop(save=True), key=lambda x: x[2] * x[3])


    #print("crops_sizeincrease",crops_person*200)

    desired_size = 200


    results.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
results.release()
cv2.destroyAllWindows()
