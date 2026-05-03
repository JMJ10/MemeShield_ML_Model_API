import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel

class ResNetBERT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.resnet = models.resnet18(weights=None)
        in_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.classifier = nn.Sequential(
            nn.Linear(in_feats + self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        img_feats = self.resnet(images)

        text_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        text_feats = text_out.pooler_output
        fused = torch.cat([img_feats, text_feats], dim=1)

        return self.classifier(fused)