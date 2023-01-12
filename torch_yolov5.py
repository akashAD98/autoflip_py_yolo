import cv2
import torch

cap = cv2.VideoCapture('/content/withbackgroung_op.mp4')
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
    results = model(frame)
    #print("1",results.pandas().xyxy[0])

    print(results.xywh)


    #crops_person = results.crop(save=True)

    #face = max(results.crop(save=True), key=lambda x: x[2] * x[3])


    #print("crops_sizeincrease",crops_person*200)

    desired_size = 200


    result.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
result.release()
cv2.destroyAllWindows()
