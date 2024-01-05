#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 18:42:02 2024

@author: becky
"""

import os
from model import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define transformations
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#load train data
data_dir = "cluster_data"
train_data = os.path.join(data_dir, "train")
# Assuming that images are now in a single subdirectory within 'train_val'
train_dataset = datasets.ImageFolder(data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

#load validation data
data_dir = "cluster_data"
train_data = os.path.join(data_dir, "val")
# Assuming that images are now in a single subdirectory within 'train_val'
valid_dataset = datasets.ImageFolder(data_dir, transform=transform)
valid_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

batch_size = 4
n_iters = 1000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

model = RoofNet()

criterion = nn.CrossEntropyLoss()

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images
        images = images.requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in valid_loader:
                # Load images
                images = images.requires_grad_()

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
 
save_model = "trained_model/model.pth"
torch.save(model.state_dict(), save_model)









