import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ThermalDataset(Dataset):
    """Dataset loader for thermal transformer images"""
    
    def __init__(self, root_dir, mode='train', transform=None, img_size=256):
        """
        Args:
            root_dir: Path to Dataset folder containing T1, T2, etc.
            mode: 'train' (only normal images) or 'test' (both faulty and normal)
            transform: Optional transforms
            img_size: Target image size
        """
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.img_size = img_size
        self.images = []
        self.labels = []
        
        # Collect all transformer folders (T1, T2, T3, T4, T5)
        transformer_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith('T')])
        
        for tf_folder in transformer_folders:
            if mode == 'train':
                # Only load normal images for training
                normal_folder = tf_folder / 'normal'
                if normal_folder.exists():
                    # Load both .jpg and .png files
                    normal_images = list(normal_folder.glob('*.jpg')) + list(normal_folder.glob('*.png'))
                    self.images.extend(normal_images)
                    self.labels.extend([0] * len(normal_images))  # 0 = normal
            else:
                # Load both faulty and normal for testing
                faulty_folder = tf_folder / 'faulty'
                normal_folder = tf_folder / 'normal'
                
                if faulty_folder.exists():
                    faulty_images = list(faulty_folder.glob('*.jpg')) + list(faulty_folder.glob('*.png'))
                    self.images.extend(faulty_images)
                    self.labels.extend([1] * len(faulty_images))  # 1 = faulty
                
                if normal_folder.exists():
                    normal_images = list(normal_folder.glob('*.jpg')) + list(normal_folder.glob('*.png'))
                    self.images.extend(normal_images)
                    self.labels.extend([0] * len(normal_images))  # 0 = normal
        
        # Default transforms
        if transform is None:
            if mode == 'train':
                self.transform = A.Compose([
                    A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(img_size, img_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
        else:
            self.transform = transform
        
        print(f"Loaded {len(self.images)} images in {mode} mode")
        if mode == 'test':
            print(f"  - Normal: {self.labels.count(0)}, Faulty: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        if self.mode == 'train':
            return image
        else:
            return image, self.labels[idx], str(img_path)
