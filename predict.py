import sys, json, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

if len(sys.argv) < 2:
    sys.exit("Usage: python predict.py path/to/image.jpg")

classes = json.load(open("weights/labels.json"))
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))
model.load_state_dict(torch.load("weights/waste_model.pth", map_location="cpu"))
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

img = Image.open(sys.argv[1]).convert("RGB")
with torch.no_grad():
    probs = torch.softmax(model(tf(img).unsqueeze(0)), 1)[0]
top = probs.topk(3)
for p, i in zip(top.values, top.indices):
    print(f"{classes[i]}: {p:.2%}")
