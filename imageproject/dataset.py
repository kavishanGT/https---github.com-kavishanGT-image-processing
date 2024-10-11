import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

class UltrasoundVideoDataset(Dataset):
    def __init__(self, frame_paths, mask_paths, transform=None):
        self.frame_paths = frame_paths  # List of paths to frames
        self.mask_paths = mask_paths    # List of paths to masks
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame = self.load_image(self.frame_paths[idx])
        frame_prev = self.load_image(self.frame_paths[max(0, idx - 1)])
        frame_next = self.load_image(self.frame_paths[min(len(self.frame_paths) - 1, idx + 1)])
        mask = self.load_mask(self.mask_paths[idx])

        if self.transform:
            frame = self.transform(frame)
            frame_prev = self.transform(frame_prev)
            frame_next = self.transform(frame_next)
            mask = self.transform(mask)

        return (frame, frame_prev, frame_next), mask

    def load_image(self, path):
        return Image.open(path).convert('RGB')

    def load_mask(self, path):
        return Image.open(path).convert('L')  # Convert to grayscale for binary mask
