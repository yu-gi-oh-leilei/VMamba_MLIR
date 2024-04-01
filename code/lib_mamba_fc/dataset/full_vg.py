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

class FullVGDataset(Dataset):
    def __init__(self, dataset_dir, img_dir, img_list, label_path, image_transform, mode):
        with open(img_list, 'r') as f:
            self.img_names = f.readlines()
        with open(label_path, 'r') as f:
            self.labels_path = json.load(f) 
        self.input_transform = image_transform
        self.dataset_dir = dataset_dir
        self.img_dir = img_dir
        self.num_labels= 500

        assert mode in ('train', 'val')

        self.mode = mode

        # with open(osp.join(dataset_dir, 'vg200_category.txt'),'r') as load_category:
        #     self.category_map = load_category.readlines()
        # self.classnames = [label.split(' ')[0].replace('\n','').replace('\r','') for label in self.category_map]


        # labels : numpy.ndarray, shape->(len(vg), 200)
        # value range->(-1 means label don't exist, 1 means label exist)
        self.labels = np.zeros((len(self.img_names), self.num_labels)).astype(np.int32)
        for i in range(len(self.img_names)):
            self.labels[i][self.labels_path[self.img_names[i][:-1]]] = 1

        self.labels = torch.from_numpy(self.labels).float()
        self.return_name=False



    def __getitem__(self, index):
        name = self.img_names[index][:-1]
        name_path = os.path.join(self.img_dir, name)
        iamge = Image.open(name_path).convert('RGB')
        
        if self.input_transform:
            iamge = self.input_transform(iamge)
        # print(len(self.labels[index]))
        if self.return_name:
            return {'image': iamge, 'target': self.labels[index], 'name': name_path}
        return {'image': iamge, 'target': self.labels[index]}

    def __len__(self):
        return len(self.img_names)


if __name__ == '__main__':
    import os.path as osp
    dataset_dir = '/media/data2/MLICdataset/'
    train_data_transform = None
    test_data_transform = None
    vg_root = osp.join(dataset_dir, 'VG')
    train_dir = osp.join(vg_root,'VG_100K')
    train_list = osp.join(vg_root,'train_list_500.txt')
    test_dir = osp.join(vg_root,'VG_100K')
    test_list = osp.join(vg_root,'test_list_500.txt')
    train_label = osp.join(vg_root,'vg_category_200_labels_index.json')
    test_label = osp.join(vg_root,'vg_category_200_labels_index.json')

    train_mode = 'train'
    val_mode = 'val'

    #                       dataset_dir, img_dir, img_list, label_path, image_transform, mode, label_proportion=1.0


    # for label_pro in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    # for label_pro in [0.6]:
    #     print(label_pro)
    train_dataset = FullVGDataset(vg_root, train_dir, train_list, train_label, train_data_transform, train_mode)
    val_dataset = FullVGDataset(vg_root, test_dir, test_list, test_label, test_data_transform, val_mode)
    print('[dataset] VG500 classification set=%s number of classes=%d  number of images=%d' % ('Train', train_dataset.num_labels, len(train_dataset)))
    print('[dataset] VG500 classification set=%s number of classes=%d  number of images=%d' % ('Test', val_dataset.num_labels, len(val_dataset)))
    print(train_dataset[0])
    print(val_dataset[1])
    # print(train_dataset[2])

    # print(train_dataset.temp.sum() / len(train_dataset))
    # print( (train_dataset.temp.sum() + val_dataset.temp.sum()) / (len(train_dataset)+ len(val_dataset))  )

        # self.temp = []
        # for index in range(len(self.img_names)):
        #     name = self.img_names[index][:-1]
        #     img_path = os.path.join(self.img_dir, name)
        #     label = np.zeros(self.num_labels).astype(np.float32)
        #     label[self.labels[name]] = 1.0
        #     self.temp.append(label)
        # self.temp = torch.from_numpy(np.array(self.temp))
