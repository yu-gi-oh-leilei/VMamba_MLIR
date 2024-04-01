import os, sys
import os.path as osp
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image

classnames = ['airport', 'animal', 'beach', 'bear', 'birds', 'boats', 'book', 'bridge', 'buildings', 
              'cars', 'castle', 'cat', 'cityscape', 'clouds', 'computer', 'coral', 'cow', 'dancing', 
              'dog', 'earthquake', 'elk', 'fire', 'fish', 'flags', 'flowers', 'food', 'fox', 'frost', 
              'garden', 'glacier', 'grass', 'harbor', 'horses', 'house', 'lake', 'leaf', 'map', 'military', 
              'moon', 'mountain', 'nighttime', 'ocean', 'person', 'plane', 'plants', 'police', 'protest', 
              'railroad', 'rainbow', 'reflection', 'road', 'rocks', 'running', 'sand', 'sign', 'sky', 
              'snow', 'soccer', 'sports', 'statue', 'street', 'sun', 'sunset', 'surf', 'swimmers', 
              'tattoo', 'temple', 'tiger', 'tower', 'town', 'toy', 'train', 'tree', 'valley', 'vehicle', 
              'water', 'waterfall', 'wedding', 'whales', 'window', 'zebra']
    

class FullNusWideDataset(Dataset):
    def __init__(self, 
        root_dir,
        img_dir,
        anno_path, 
        labels_path,
        mode='train',
        transform=None,
        rm_no_label_data=True
        ) -> None:
        """[summary]
        Args:
            img_dir ([type]): dir of imgs
            anno_path ([type]): list of used imgs
            labels_path ([type]): labels of used imgs
            transform ([type], optional): [description]. Defaults to None.
            """
        super().__init__()


        self.root_dir = root_dir
        self.img_dir = img_dir
        self.anno_path = anno_path
        self.labels_path = labels_path
        self.transform = transform
        self.rm_no_label_data = rm_no_label_data
        self.classnames = classnames
        self.mode = mode
        self.returen_name = True

        self.imgnamelist, self.labellist = self.preprocess() # [imgpath] [label]
        self.labellist = torch.tensor(np.array(self.labellist))
        # self.labellist[self.labellist==0] = -1


    def preprocess(self):
        imgnamelist = [line.strip().replace('\\', '/') for line in open(self.anno_path, 'r')]
        labellist = [line.strip() for line in open(self.labels_path, 'r')]
        assert len(imgnamelist) == len(labellist)
        
        res = []
        imgname_list = []
        label_list = []

        for idx, (imgname, labelline) in enumerate(zip(imgnamelist, labellist)):
            imgpath = osp.join(self.img_dir, imgname)
            labels = [int(i) for i in labelline.split(' ')]
            labels = np.array(labels).astype(np.float32)
            if sum(labels) == 0:
                continue

            imgname_list.append(imgpath)
            label_list.append(labels)
            # label_list.append(torch.tensor(labels))


        return imgname_list, label_list
    
    def __len__(self) -> int:
        return len(self.imgnamelist)

    def __getitem__(self, index: int):
        imgpath = self.imgnamelist[index]

        img = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.returen_name:
            return {'image': img, 'target': self.labellist[index], 'name': imgpath}
        else:
            return {'image': img, 'target': self.labellist[index]}



# /media/data/maleilei/MLICdataset/nus_wide
if __name__ == '__main__':
    # train_label = osp.join(vg_root, 'NUS_WID_Tags','Train_Tags81.txt')
    # test_label = osp.join(vg_root, 'NUS_WID_Tags','Test_Tags81.txt')
    ds = FullNusWideDataset(
        # img_dir = '/media/data/maleilei/MLICdataset/nus_wide/Flickr',
        # img_dir = '/media/data/maleilei/MLICdataset/NUS-WIDE-downloader/image',
        # /media/data/maleilei/MLICdataset/nuswide/slsplit
        root_dir = '/media/data2/MLICdataset/nuswide',
        img_dir = '/media/data2/MLICdataset/nuswide/Flickr',
        anno_path = '/media/data2/MLICdataset/nuswide/ImageList/TrainImagelist.txt',
        labels_path = '/media/data2/MLICdataset/nuswide/Groundtruth/Labels_Train.txt',
        mode='train',
    )
    print(ds[1])
    # /media/data/maleilei/MLICdataset/nuswide/slsplit
    # /media/data/maleilei/MLICdataset/nuswide/Groundtruth

    ds_test = FullNusWideDataset(
        # img_dir = '/media/data/maleilei/MLICdataset/nus_wide/Flickr',
        # img_dir = '/media/data/maleilei/MLICdataset/NUS-WIDE-downloader/image',
        root_dir = '/media/data2/MLICdataset/nuswide',
        img_dir = '/media/data2/MLICdataset/nuswide/Flickr',
        anno_path = '/media/data2/MLICdataset/nuswide/ImageList/TestImagelist.txt',
        labels_path = '/media/data2/MLICdataset/nuswide/Groundtruth/Labels_Test.txt',
    )
    print(ds_test[1])
    # # # Test_label.txt  Test_split.txt  Train_label.txt  Train_split.txt
    # # # Labels_Train.txt Labels_Test.txt
    # print("len(ds):", len(ds)) 
    # print("len(ds_test):", len(ds_test))