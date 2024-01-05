import glob
import numpy as np
import torch
import albumentations as A

# from utils import get_label_mask, set_class_values
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SolarDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        # self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, self.images[index])
        # mask_path = os.path.join(self.mask_dir, self.images[index].replace(".bmp", "_label.bmp"))
        image = Image.open(img_path).convert("RGB")
        # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # mask[mask == 255.0] = 1.0
        
        # sample = {'image': img_name, 'landmarks': img_name}
        if self.transform is not None:
            image = self.transform(image)
        #     image = augmented['image']
        #     # mask = augmented['mask']

        return image, img_name
    
# if __name__ == "__main__":
    
#     path_img = "data/images"
#     path_mask = "data/masks"
#     transform = A.Compose([
#         A.Resize(256, 256),
#         A.HorizontalFlip(p=0.5),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ])
    
#     dataset = SolarDataset(image_dir=path_img, mask_dir=path_mask, transform=transform)
#     from torch.utils.data import DataLoader

#     data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    

