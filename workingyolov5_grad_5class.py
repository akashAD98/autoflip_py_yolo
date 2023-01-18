import torch

from models.common import DetectMultiBackend

#model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()


weights='/content/drive/MyDrive/yolov5/runs/train-cls/exp9/weights/best.pt'

model = DetectMultiBackend(weights)
print("model loadeddd")

import requests
from PIL import Image
from torchvision import transforms

# Download human-readable labels for ImageNet.
#response = requests.get("https://git.io/JJkYN")
labels = '/content/classes.txt'
#response.text.split("\n")


def predict(inp):
  inp = transforms.ToTensor()(inp).unsqueeze(0)
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(5)}    
  return confidences



import gradio as gr

gr.Interface(fn=predict, 
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=3)).launch(debug=True)
