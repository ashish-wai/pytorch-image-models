import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

class ImageDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.labels2id = {}
        self.id2label = {}
        for class_name in sorted(os.listdir(os.path.join(root_dir, split))):
            class_dir = os.path.join(root_dir, split, class_name)
            for image_name in sorted(os.listdir(class_dir)):
                image_path = os.path.join(class_dir, image_name)
                label = class_name  
                self.image_paths.append(image_path)
                self.labels.append(label)
        self.unique_labels = list(set(self.labels))
        self.unique_labels.sort()
        for idx, label in enumerate(self.unique_labels):
            self.labels2id[label] = idx
            self.id2label[idx] = label
        
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # if image is heic
        if image_path.endswith('.HEIC') or image_path.endswith('.heic'):
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        label = self.labels2id[self.labels[idx]]
        if self.transform:
            image = self.transform(image)
        # return {'image_file_path': image_path, 'image': image, 'labels': label}
        return image, label