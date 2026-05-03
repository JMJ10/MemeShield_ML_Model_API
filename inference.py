import torch
import json
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from model import ResNetBERT
from reasoning import reasoning_layer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("weights/label_map.json") as f:
    label_map = json.load(f)

idx2label = {v: k for k, v in label_map.items()}


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = ResNetBERT(num_classes=len(label_map))

MODEL_PATH="weights/fusion_inference_only.pth"
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    import gdown
    url = "https://drive.google.com/uc?id=1W-udpxu9GOZTaBRldJ5ResXyiMyNCqNM"
    os.makedirs("weights", exist_ok=True)
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Download complete.")

model.load_state_dict(torch.load("weights/fusion_inference_only.pth", map_location=device))
model.to(device).eval()


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def predict(image_pil, text=""):
    with torch.no_grad():

        img_tensor = transform(image_pil).unsqueeze(0).to(device)

        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(device)

        logits = model(
            img_tensor,
            tokens["input_ids"],
            tokens["attention_mask"]
        )

        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs))

        result =  {
            "label": idx2label[pred_idx],
            "score": float(probs[pred_idx])
        }

        if result["label"] == "offensive":
            reasoning = reasoning_layer(
                text,
                result,
                tokenizer,
                model.bert,
                device
            )
            result.update(reasoning)
        else:
            result.update({
                "categories": [],
                "laws": [],
                "explanation": "Content does not appear offensive."
            })

        return result