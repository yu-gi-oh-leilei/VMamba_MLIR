import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from pdb import set_trace as stop
import json
from PIL import Image
import time

import os.path as osp

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def image_loader(path,transform):
    try:
        image = Image.open(path)
    except FileNotFoundError: # weird issues with loading images on our servers
        # print('FILE NOT FOUND')
        time.sleep(10)
        image = Image.open(path)

    image = image.convert('RGB')

    if transform is not None:
        image = transform(image)

    return image

class FullCUB200Dataset(Dataset):
    def __init__(self, dataset_dir, img_dir, image_transform, mode):


        self.input_transform = image_transform
        self.dataset_dir = dataset_dir
        self.img_dir = img_dir
        self.num_labels= 312

        assert mode in ('train', 'val')

        self.mode = mode

        if mode == 'train':
            self.img_names = np.load(os.path.join(dataset_dir, 'full_train_images.npy'))
            self.labels = np.load(os.path.join(dataset_dir, 'full_train_labels.npy'))
        elif mode == 'val':
            self.img_names = np.load(os.path.join(dataset_dir, 'full_val_images.npy'))
            self.labels = np.load(os.path.join(dataset_dir, 'full_val_labels.npy'))
        
        self.labels = torch.from_numpy(self.labels).float()
        self.return_name = True


    def __getitem__(self, index):
        name = self.img_names[index]
        name_path = osp.join(self.img_dir, name)
        iamge = Image.open(name_path).convert('RGB')
        
        if self.input_transform:
            iamge = self.input_transform(iamge)
        if self.return_name:
            return {'image': iamge, 'target': self.labels[index], 'name': name_path}
        return {'image': iamge, 'target': self.labels[index]}

    def __len__(self):
        return len(self.img_names)

def prepare_full_cub200():

    import os
    import json
    import numpy as np
    import argparse
    import pandas as pd

    load_path = '/media/data2/MLICdataset/CUB_200_2011'
    save_path = '/media/data2/MLICdataset/CUB_200_2011'

    NUM_ATTRIBUTES = 312
    NUM_INSTANCES = 11788

    images_df = pd.read_csv(
        os.path.join(load_path, 'CUB_200_2011', 'images.txt'),
        delimiter = ' ',
        header = None,
        names = ['index', 'filename'],
        usecols = ['filename']
        )
    images_np = images_df.to_numpy()
    assert len(images_np) == NUM_INSTANCES

    # get splits:
    splits_df = pd.read_csv(
        os.path.join(load_path, 'CUB_200_2011', 'train_test_split.txt'),
        delimiter = ' ',
        header = None,
        names = ['index', 'is_train'],
        usecols = ['is_train']
        )
    splits_np = splits_df.to_numpy()

    # get classes:
    classes_df = pd.read_csv(
        os.path.join(load_path, 'attributes.txt'),
        delimiter = ' ',
        header = None,
        names = ['index', 'attribute_name'],
        usecols = ['attribute_name']
        )
    classes_np = classes_df.to_numpy()
    assert len(classes_np) == 312

    # get labels:
    attributes_df = pd.read_csv(
        os.path.join(load_path, 'CUB_200_2011', 'attributes', 'image_attribute_labels.txt'),
        delimiter = ' ',
        header = None,
        names = ['image_id', 'attribute_id', 'is_present', 'certainty_id', 'time'],
        usecols = ['attribute_id', 'is_present']
        )
    attributes_np = attributes_df.to_numpy()
    assert len(attributes_np) == NUM_ATTRIBUTES * NUM_INSTANCES

    labels_train = []
    images_train = []
    labels_test = []
    images_test = []
    k = 0
    for i in range(NUM_INSTANCES):
        label_vector = []
        for j in range(NUM_ATTRIBUTES):
            label_vector.append(int(attributes_np[k, 1]))
            k += 1
        if splits_np[i] == 1:
            labels_train.append(label_vector)
            images_train.append(str(images_np[i][0]))
        else:
            labels_test.append(label_vector)
            images_test.append(str(images_np[i][0]))


    np.save(os.path.join(save_path, 'full_train_labels.npy'), np.array(labels_train))
    np.save(os.path.join(save_path, 'full_train_images.npy'), np.array(images_train))
    np.save(os.path.join(save_path, 'full_val_labels.npy'), np.array(labels_test))
    np.save(os.path.join(save_path, 'full_val_images.npy'), np.array(images_test))

if __name__ == '__main__':
    # prepare_full_cub200()
    # 
    save_path = '/media/data2/MLICdataset/CUB_200_2011'
    import numpy as np
    full_train_images = np.load(os.path.join(save_path, 'full_train_images.npy'))
    full_train_labels = np.load(os.path.join(save_path, 'full_train_labels.npy'))

    print(full_train_images[0])
    print(full_train_labels[0])
    # /media/data2/MLICdataset/CUB_200_2011/CUB_200_2011/images
    dataset_dir = '/media/data2/MLICdataset/CUB_200_2011'
    img_dir = os.path.join(dataset_dir, 'CUB_200_2011', 'images')
    image_transform = None
    mode = 'train'
    train_dataset = FullCUB200Dataset(dataset_dir, img_dir, image_transform, mode)
    print(train_dataset[0])

    mode = 'val'
    test_dataset = FullCUB200Dataset(dataset_dir, img_dir, image_transform, mode)

    print(test_dataset[0])
