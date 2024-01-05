#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 19:40:47 2024

@author: becky
"""

import torch
import argparse
import cv2
import csv
import os

from PIL import Image
from model import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

save_model = "trained_model/model.pth"

# Define transformations
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#load train data
data_dir = "test_bmp_only_jpg"
# # Assuming that images are now in a single subdirectory within 'train_val'
# test_dataset = datasets.ImageFolder(data_dir, transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load
model = RoofNet()
model.load_state_dict(torch.load(save_model))
model.eval()

data = []
data.append(['file_ name', 'roof type'])
for img_file in os.listdir(data_dir):
    image = Image.open(os.path.join(data_dir, img_file)).convert("RGB")#images.requires_grad_()
    
    img = transform(image)
    # pass the image to the model
    outputs = model(img.unsqueeze(0))

    # Get predictions from the maximum value
    _, predicted = torch.max(outputs.data, 1)
    
    data.append([img_file, int(predicted[0])])

filename = 'result.csv'
# Writing the result to the CSV file
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    # Writing each row
    for row in data:
        writer.writerow(row)


















