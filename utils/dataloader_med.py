import copy
import os
import random
import numpy as np
import torchvision.transforms.functional
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import torch
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import Dataset


class ChestX_ray14(Dataset):
    def __init__(self, data_dir, file, augment,
                 num_class=14, img_depth=3, heatmap_path=None,
                 data_pct=1, seed=0, mode='train',
                 pretraining=False):
        self.img_list = []
        self.img_label = []

        with open(file, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(data_dir, lineItems[0])
                    imageLabel = lineItems[1:num_class + 1]
                    imageLabel = [int(i) for i in imageLabel]
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

        self.augment = augment
        self.img_depth = img_depth
        if heatmap_path is not None:
            self.heatmap = Image.open(heatmap_path).convert('RGB')
        else:
            self.heatmap = None
        self.pretraining = pretraining

        if data_pct < 1.0 and data_pct > 0.0 and mode == 'train':
            random.seed(seed)
            index = random.sample(range(len(self.img_list)), int(len(self.img_list) * data_pct))
            self.img_list = [self.img_list[i] for i in index]
            self.img_label = [self.img_label[i] for i in index]

        self.mode = mode
        self.pct = data_pct
        if self.pct < 1 and mode == 'train':
            self.true_len = len(self.img_list)
            self.merge_len = int(self.true_len / self.pct)


    def __len__(self):
        if self.pct < 1 and self.mode == 'train':
            return self.merge_len
        return len(self.img_list)

    def __getitem__(self, index):

        if self.pct < 1 and self.mode == 'train':
            index = index % self.true_len

        file = self.img_list[index]
        label = self.img_label[index]

        imageData = Image.open(file).convert('RGB')
        if self.heatmap is None:
            imageData = self.augment(imageData)
            img = imageData
            label = torch.tensor(label, dtype=torch.float)
            if self.pretraining:
                label = -1
            return img, label
        else:
            heatmap = self.heatmap
            imageData, heatmap = self.augment(imageData, heatmap)
            img = imageData
            heatmap = heatmap.permute(1, 2, 0)
            label = torch.tensor(label, dtype=torch.float)
            if self.pretraining:
                label = -1
            return [img, heatmap], label
