#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:44:59 2024

@author: becky
"""
import os
import cv2
import torch
import shutil
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from datasets import *

# Function to extract features
def extract_features(model, dataloader):
    features = []
    with torch.no_grad():  # No need to calculate gradients
        for inputs, _ in dataloader:
            outputs = model(inputs)
            features.append(outputs)
    return torch.cat(features, dim=0)

def plot_images_from_cluster(cluster_id, n_images, images, clusters):
    plt.figure(figsize=(10, 10))
    images_from_cluster = np.where(clusters == cluster_id)[0]
    for i, idx in enumerate(images_from_cluster[:n_images]):
        plt.subplot(n_images // 5 + 1, 5, i + 1)
        plt.imshow(images[idx])
        plt.title(f"Cluster {cluster_id}")
        plt.axis('off')
    plt.show()
    

# Assuming images are stored in a directory named 'train_val'
# train_val = "data/"

path_img = "data_jpg/images_jpg"
    

# Define transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = SolarDataset(image_dir=path_img, transform=transform)

# dataset = datasets.ImageFolder(train_val, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load pre-trained VGG16 model
model = models.vgg16(pretrained=True).eval()  # Set model to evaluation mode

# Remove the last layer (classifier) to use it as a feature extractor
model.classifier = model.classifier[:-1]

# Extract features
features = extract_features(model, dataloader)

pca = PCA(n_components=50)  # Reduce to 50 dimensions
reduced_features = pca.fit_transform(features.reshape(features.shape[0], -1))

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=0)  # Example with 5 clusters
clusters = kmeans.fit_predict(reduced_features)

numpy_arrays = []
all_image_name = []
all_image = []
for batch in dataloader:
    np_image = batch[0].cpu().numpy().squeeze(0)
    # Convert tensor to numpy array after moving to CPU
    numpy_arrays.append(np_image)
    all_image_name.append(batch[1][0])
    all_image.append(cv2.imread(os.path.join(path_img, batch[1][0])))

data_dir = "cluster_data"

for i in range(n_clusters):
    class_name = "cluster_" + str(i)
    path = os.path.join(data_dir, class_name)
    if not os.path.exists(path):
        os.makedirs(path)
        
    
# # Sample usage: plot 10 images from cluster 0
plot_images_from_cluster(0, 10, all_image, clusters)

# data_dir = "cluster_data"
for cluster, image_file in zip(clusters, all_image_name):
    if cluster==0:
        cluster_path = os.path.join(data_dir, 'cluster_0')
        src = os.path.join(path_img, image_file)
        dst = os.path.join(data_dir, 'cluster_0', image_file)        
        shutil.copy(src, dst)
    if cluster==1:
        cluster_path = os.path.join(data_dir, 'cluster_1')
        src = os.path.join(path_img, image_file)
        dst = os.path.join(data_dir, 'cluster_1', image_file)        
        shutil.copy(src, dst)
    if cluster==2:
        cluster_path = os.path.join(data_dir, 'cluster_2')
        src = os.path.join(path_img, image_file)
        dst = os.path.join(data_dir, 'cluster_2', image_file)        
        shutil.copy(src, dst)
    if cluster==3:
        cluster_path = os.path.join(data_dir, 'cluster_3')
        src = os.path.join(path_img, image_file)
        dst = os.path.join(data_dir, 'cluster_3', image_file)        
        shutil.copy(src, dst)
    if cluster==4:
        cluster_path = os.path.join(data_dir, 'cluster_4')
        src = os.path.join(path_img, image_file)
        dst = os.path.join(data_dir, 'cluster_4', image_file)        
        shutil.copy(src, dst)
    
    












