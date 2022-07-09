import numpy as np
import pandas as pd
import altair as alt

from pathlib import Path
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms as T

from torchinfo import summary

dev = torch.device(('cpu', 'cuda')[torch.cuda.is_available()])

# dataset

img_url, top_url = ['/'.join([
    'https:/',
    'raw.githubusercontent.com',
    'Morshay',
    'tag-ur-it',
    'main',
    f'{f}_tags.csv'
]) for f in ['img', 'top']]

all_labels = pd.read_csv(img_url, converters={'tags': eval})
label_converter = pd.read_csv(top_url).squeeze()

# helper defs for dataset

def lbls2proba(labels):
    return torch.FloatTensor(
        label_converter.apply(
            lambda name:
            .9 if name in labels else .1
        )
    )

preprocess = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# dataset class

class DanbooruDataset(Dataset):

    def __init__(self, label_data, img_dir,
                 transform=preprocess,
                 target_transform=lbls2proba):
        
        self.label_data = label_data
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        img_id = self.label_data.iloc[idx, 0]
        img_path = Path(self.img_dir) / f'{img_id}.jpg'
        image = self.transform(Image.open(img_path).convert('RGB'))        
        labels = self.target_transform(self.label_data.iloc[idx, 1])
        return image.to(dev), labels.to(dev)

# output layers on top of effnet

def bn_drop_lin(in_size, out_size):
    return nn.Sequential(
        nn.BatchNorm1d(
            in_size,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True),
        nn.Dropout(p=0.25, inplace=False),
        nn.Linear(in_size, out_size)
    )

# model class

class EffnetTagger(nn.Module):
    def __init__(self,
                 out_classes=len(label_converter),
                 base_model='efficientnet_b4',  # effnet_v2_s SOON™
                 effnet_out_features=1792):
        super(EffnetTagger, self).__init__()

        self.out_classes = out_classes
        self.effnet_out_features = effnet_out_features

        net = torch.hub.load('pytorch/vision:v0.12.0',
                             base_model, pretrained=True)

        self.effnet = nn.Sequential(*list(net.children())[:-1])
        self.out_1 = bn_drop_lin(self.effnet_out_features, 512)
        self.out_2 = bn_drop_lin(512, self.out_classes)

    def forward(self, t_in):

        t = F.leaky_relu(self.effnet(t_in))[:, :, 0, 0]

        t1 = F.leaky_relu(self.out_1(t))
        t2 = self.out_2(t1)

        t_rs = t2.reshape([len(t), self.out_classes])
        t_cl = torch.clamp(t_rs, -10, 10)

        t_out = torch.sigmoid(t_cl)

        return t_out