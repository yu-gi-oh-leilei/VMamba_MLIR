import os
import sys
# sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..', 'cocoapi/PythonAPI'))
# sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.datasets as datasets

from pycocotools.coco import COCO
import os.path as osp

category_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42, "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}

classnames = ['person','bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                    'toothbrush']

class FullCOCO2017dataset(data.Dataset):

    def __init__(self, dataset_dir, 
                mode, image_dir, anno_path, labels_path=None,
                input_transform=None):

        assert mode in ('train', 'val')

        self.dataset_dir = dataset_dir
        self.mode = mode
        self.labels_path = labels_path
        self.input_transform = input_transform
        self.classnames = classnames
        self.classnames = [syn[0] for syn in coco_classname_synonyms]
        
        self.root = image_dir
        self.coco = COCO(anno_path)
        self.ids = list(self.coco.imgs.keys())

        with open(osp.join(dataset_dir, 'data', 'category.json'),'r') as load_category:
            self.category_map = json.load(load_category)

        os.makedirs(osp.join(dataset_dir, 'data'), exist_ok=True)
        label_path = osp.join(dataset_dir, 'data', '{}_label_vectors_coco17.npy'.format(mode))
        
        if os.path.exists(label_path):
            print(label_path)
            self.labels = np.load(label_path)
        else:
            self.labels = []
            for i in range(len(self.ids)):
                img_id = self.ids[i]
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                target = self.coco.loadAnns(ann_ids)
                self.labels.append(getLabelVector(getCategoryList(target), self.category_map))
            self.labels = np.array(self.labels)
            np.save(label_path, self.labels)

            print(label_path)
        


        self.labels = torch.from_numpy(self.labels)

        self.return_name = True


    def __getitem__(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        input = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.input_transform:
            input = self.input_transform(input)

        if self.return_name:
            name = path
            return {'image': input, 'target': self.labels[index], 'name': name}
        return {'image': input, 'target': self.labels[index]}

    def __len__(self):
        return len(self.ids)

# =============================================================================
# Help Functions
# =============================================================================
def getCategoryList(item):
    categories = set()
    for t in item:
        categories.add(t['category_id'])
    return list(categories)

def getLabelVector(categories, category_map):
    label = np.zeros(80)
    for c in categories:
        label[category_map[str(c)]-1] = 1.0
    return label


coco_classname_synonyms = [
    ['person', 'human', 'people', 'man', 'woman', 'passenger'], 
    ['bicycle', 'bike', 'cycle'],
    ['car', 'taxi', 'auto', 'automobile', 'motor car'], 
    ['motor bike', 'motor cycle'], 
    ['aeroplane', "air craft", "jet", "plane", "air plane"], 
    ['bus', 'autobus', 'coach', 'charabanc', 'double decker', 'jitney', 'motor bus', 'motor coach', 'omnibus'],
    ['train', 'rail way', 'railroad'], 
    ['truck'],
    ['boat', 'raft', 'dinghy'],
    ['traffic light'],
    ['fire hydrant', 'fire tap', 'hydrant'],
    ['stop sign', 'halt sign'],
    ['parking meter'],
    ['bench'],
    ['bird'],
    ['cat', 'kitty'],
    ['dog', 'pup', 'puppy', 'doggy'],
    ['horse', 'colt', 'equus'],
    ['sheep'],
    ['cow'],
    ['elephant'],
    ['bear'],
    ['zebra'],
    ['giraffe', 'camelopard'],
    ['backpack', 'back pack', 'knapsack', 'packsack', 'rucksack', 'haversack'],
    ['umbrella'],
    ['handbag', 'hand bag', 'pocketbook', 'purse'],
    ['tie', 'necktie'],
    ['suitcase'],
    ['frisbee'],
    ['skis', 'ski'],
    ['snowboard'],
    ['sports ball', 'sport ball', 'ball', 'football', 'soccer', 'tennis', 'basketball', 'baseball'],
    ['kite'],
    ['baseball bat', 'baseball game bat'],
    ['baseball glove', 'baseball mitt', 'baseball game glove'],
    ['skateboard'],
    ['surfboard'],
    ['tennis racket'],
    ['bottle'],
    ['wine glass', 'vino glass'],
    ['cup'],
    ['fork'],
    ['knife'],
    ['spoon'],
    ['bowl'],
    ['banana'],
    ['apple'],
    ['sandwich'],
    ['orange'],
    ['broccoli'],
    ['carrot'],
    ['hot dog'],
    ['pizza'],
    ['donut', 'doughnut'],
    ['cake'],
    ['chair', 'arm chair'],
    ['couch', 'sofa'],
    ['potted plant', 'house plant', 'bonsai', 'pot plant'],
    ['bed'],
    ['dining table', 'dinner table', 'table', 'din table'], 
    ['toilet', 'commode'],
    ['tv', 'tvmonitor', 'monitor', 'television', 'telly'],
    ['laptop'],
    ['mouse'],
    ['remote'],
    ['keyboard'],
    ['cell phone', 'phone', 'mobile phone'],
    ['microwave'],
    ['oven', 'roaster'],
    ['toaster'],
    ['sink'],
    ['refrigerator', 'icebox'],
    ['book'],
    ['clock'],
    ['vase'],
    ['scissors'],
    ['teddy bear', 'teddy'],
    ['hair drier', 'blowing machine', 'hair dryer', 'dryer', 'blow dryer', 'blown dry', 'blow dry'],
    ['toothbrush'],
]



if __name__ == '__main__':
    import os.path as osp
    prob = 1.0
    train_data_transform = None
    test_data_transform = None


    dataset_dir = '/media/data2/MLICdataset/'
    dataset_dir = osp.join(dataset_dir, 'COCO2017')

    train_dir = osp.join(dataset_dir, 'train2017')
    train_anno_path = osp.join(dataset_dir, 'annotations/instances_train2017.json')
    train_label_path = osp.join(dataset_dir, 'data/train_label_vectors_coco17.npy')
    
    
    test_dir = osp.join(dataset_dir, 'val2017')
    test_anno_path = osp.join(dataset_dir, 'annotations/instances_val2017.json')
    test_label_path = osp.join(dataset_dir, 'data/val_label_vectors_coco1.npy')

    

    train_dataset = FullCOCO2017dataset(
            dataset_dir=dataset_dir,
            mode='train',
            image_dir=train_dir,
            anno_path=train_anno_path,
            labels_path=train_label_path,
            input_transform=train_data_transform,
        )
    val_dataset = FullCOCO2017dataset(
            dataset_dir=dataset_dir,
            mode='val',
            image_dir=test_dir,
            anno_path=test_anno_path,
            labels_path=test_label_path,
            input_transform=test_data_transform
        )

    print(train_dataset[0])
    print(val_dataset[0])


    # path_part = '/media/data/maleilei/MLICdataset/COCO2014/part_coco_detection/label_proportion_1.0/train_label_vectors_1.0.npy'
    # path_q2l = '/media/data/maleilei/MLICdataset/COCO2014/label_npy/train_label_vectors_coco14.npy'

    # # [1, -1] => [1, 0]
    # path_part = np.load(path_part)
    # path_part[path_part==-1] = 0
    # target_part = torch.from_numpy(path_part)

    # # [1, 0]
    # path_q2l = np.load(path_q2l)
    # target_q2l = torch.from_numpy(path_q2l)

    # print((target_part - target_q2l).sum())
