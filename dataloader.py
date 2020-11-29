import torch
import os
from PIL import Image


def get_image_list(raw_image_path, clear_image_path, is_train):
    image_list = []
    raw_image_list = [raw_image_path + i for i in os.listdir(raw_image_path)]
    if is_train:
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image, os.path.join(clear_image_path + image_file), image_file])
    else:
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image, None, image_file])
    return image_list


class UWNetDataSet(torch.utils.data.Dataset):
    def __init__(self, raw_image_path, clear_image_path, transform, is_train=False):
        self.raw_image_path = raw_image_path
        self.clear_image_path = clear_image_path
        self.is_train = is_train
        self.image_list = get_image_list(self.raw_image_path, self.clear_image_path, is_train)
        self.transform = transform

    def __getitem__(self, index):
        raw_image, clear_image, image_name = self.image_list[index]
        raw_image = Image.open(raw_image)
        if self.is_train:
            clear_image = Image.open(clear_image)
            return self.transform(raw_image), self.transform(clear_image), "_"
        return self.transform(raw_image), "_", image_name

    def __len__(self):
        return len(self.image_list)
