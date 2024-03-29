#%%

import re
import numpy as np
import os
import cv2
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from torch.utils.data import DataLoader

# regex: [^\(]\w*[^\)]
#%%
class FlowerLoader(Dataset):
    def __init__(self, image_folder_path):
        super(FlowerLoader).__init__()
        self.image_names = os.listdir(image_folder_path)
        self.image_paths = [f"{image_folder_path}/{i}" for i in self.image_names]
        
        # Image transformation
        self.transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.CenterCrop((224, 224)),
            transforms.Normalize(
                mean= (0.4583, 0.4191, 0.2997), 
                std=(0.2962, 0.2650, 0.2880)
            )
        ])
    
    def label_translator(self, label):
        INDEXES = {0: 'Daisy', 1: 'Rose', 2: 'Tulip', 3: 'Dandelion', 4: 'Sunflower'}
        for i, flower in INDEXES.items():
            if label == flower.lower():
                return i
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,index):
        img = cv2.imread(self.image_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transformation(img)
        
        label = re.search(r"[^\(]\w*[^\)]", Path(self.image_paths[index]).stem)
        label = self.label_translator(label=label[0])
        
        return (img, label)
# %%
# x = torch.stack([img for img, _ in data],dim=3)
# print(x.shape)
# x = x.view(3, -1)
# print(x.mean(dim=1))
# print(x.std(dim=1))