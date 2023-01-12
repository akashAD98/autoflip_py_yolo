import cv2
import torch

cap = cv2.VideoCapture('/content/withbackgroung_op.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0]

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
videoWriter = cv2.VideoWriter("/content/op/op_newsss.mp4", fourcc, fps, size)

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
            
            # https://stackoverflow.com/questions/56115874/how-to-convert-bounding-box-x1-y1-x2-y2-to-yolo-style-x-y-w-h
            
            dw = 1./frame[0]
            dh = 1./frame[1]
            print('width',dw)
            print('height',dh)
            x = (int(pred[0]) +int(pred[1]))/2.0
            y = (int(pred[2]) + int(pred[3]))/2.0
            w = int(pred[1]) -int(pred[0])
            h = int(pred[3]) -int(pred[2])
            x = x*dw
            w = w*dw
            y = y*dh
            h = h*dh

            print("w",w)
            print("h",h)

            score = pred[4]
            #print("score",score)
            category_name = model.names[int(pred[5])]
            #print('category_name',category_name)
            category_id = pred[5]
            #print('category_id',category_id)
            
            Pesron_size = 200
            aspect_ratio = (w / h)
            print("aspect_ratio",aspect_ratio)

            # Calculate the width and height of the resized frame
            ##new_width = int(Pesron_size * aspect_ratio)
            ##new_height = int(Pesron_size / aspect_ratio)
            ##frame = cv2.resize(frame, (new_width, new_height))
        #out.write(frame)

    # Show the frame
    #cv2.imshow("Frame", frame)
    
    #crops_person = results.crop(save=True)

    #face = max(results.crop(save=True), key=lambda x: x[2] * x[3])
    #print("crops_sizeincrease",crops_person*200)
    results.show()

    #results.save()
    #results.save("/content/op")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
videoWriter.release()
cv2.destroyAllWindows()
