import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FineTuneThermalDataset(Dataset):
    """
    Dataset for fine-tuning: includes all images from the latest folder (by month-year),
    and 2x as many randomly sampled images from all other folders (base + previous fetches).
    """
    def __init__(self, root_dir, img_size=256, transform=None, latest_folder=None, old_multiplier=5, seed=42):
        """
        Args:
            root_dir: Path to Local_Dataset (contains month folders and Base_Dataset)
            img_size: Target image size
            transform: Optional albumentations transform
            latest_folder: Optionally specify the latest folder (else auto-detect)
            seed: Random seed for reproducibility
        """
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.images = []
        self.labels = []
        random.seed(seed)

        # Find all month folders (e.g., 05_2025, 07_2025) and base
        all_folders = [d for d in self.root_dir.iterdir() if d.is_dir()]
        month_folders = [d for d in all_folders if '_' in d.name and d.name[:2].isdigit()]
        base_folders = [d for d in all_folders if d.name.lower().startswith('base')]

        # Auto-detect latest folder if not provided
        if latest_folder is None:
            def folder_key(f):
                try:
                    m, y = f.name.split('_')
                    return int(y)*100 + int(m)
                except:
                    return 0
            latest_folder = max(month_folders, key=folder_key)
        else:
            latest_folder = self.root_dir / latest_folder
        print(f"Latest folder: {latest_folder}")

        # Helper to get all images from a folder (normal only)
        def get_normal_images(folder):
            normal_folder = folder / 'normal'
            if normal_folder.exists():
                return list(normal_folder.glob('*.jpg')) + list(normal_folder.glob('*.png'))
            return []

        # Special: get all normal images from Base_Dataset/T*/normal
        def get_base_normal_images(base_folder):
            images = []
            if not base_folder.exists():
                return images
            for tf_folder in base_folder.iterdir():
                if tf_folder.is_dir() and tf_folder.name.startswith('T'):
                    images.extend(get_normal_images(tf_folder))
            return images

        # 1. All images from latest folder
        latest_images = get_normal_images(latest_folder)
        N = len(latest_images)
        self.images.extend(latest_images)
        self.labels.extend([0]*N)
        self.latest_count = N
        self.latest_folder_name = latest_folder.name

        # 2. All other folders (base + previous months)
        other_folders = [f for f in (month_folders + base_folders) if f != latest_folder]
        other_images = []
        folder_image_counts = {self.latest_folder_name: N}
        for folder in other_folders:
            if folder.name.lower().startswith('base'):
                imgs = get_base_normal_images(folder)
            else:
                imgs = get_normal_images(folder)
            other_images.extend(imgs)
            folder_image_counts[folder.name] = len(imgs)
        # Randomly sample old_multiplier*N from other images
        sample_size = min(old_multiplier*N, len(other_images))
        sampled_other = random.sample(other_images, sample_size) if sample_size > 0 else []
        self.images.extend(sampled_other)
        self.labels.extend([0]*len(sampled_other))
        self.old_count = len(sampled_other)
        self.folder_image_counts = folder_image_counts

        # Shuffle all
        combined = list(zip(self.images, self.labels))
        random.shuffle(combined)
        # Print before unpacking (since self.images is a list here)
        print(f"Loaded {len(self.images)} images for fine-tuning (N={N} latest, {len(sampled_other)} sampled old, multiplier={old_multiplier})")
        self.images, self.labels = zip(*combined) if combined else ([],[])

        # Default transforms
        if transform is None:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        return image
