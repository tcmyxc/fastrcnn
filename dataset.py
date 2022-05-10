import os

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO

from roi_extraction import selective_search_extraction
from utils import bbox_resize

"""
Main coordinates of bbox is x,y,w,h
"""


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='/data/coco', data_type='train', data_size='full',
                 input_size=(512, 512), cat_type='full', train=True, ss_roi=True):
        super().__init__()

        assert len(input_size) == 2, "Dimension of input size should be 2."
        assert data_size == 'full' or type(data_size) == int

        self.input_size = input_size
        self.ss_roi = ss_roi

        self.data_path = os.path.join(data_dir, "images", data_type)

        self.cocotool, self.imgIds = self._init_cocotools(data_dir, data_type, data_size, train)

        self.category = self.cocotool.cats
        self.numbered_category = [0] + list(self.category.keys())  # 缺少 12, 26, 29, 30, 45, 66, 68, 69, 71, 83
        # print(self.numbered_category)

        self._rescale_bboxes(input_size)

        self.target_means, self.target_stds = self._target_normalize_coef()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=input_size),
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, index):
        """
        :param index:
        :return: img, anns, extracted_region
        img : normalized image tensor
        anns : corresponding annotations (bbox, class)
        extarcted_regions : region porposals by selective search, if self.ss_roi == False then return None
        """
        img_id = self.imgIds[index]
        ann_ids = self.cocotool.getAnnIds(img_id)

        filename = self.cocotool.imgs[img_id]['file_name']
        path = os.path.join(self.data_path, filename)
        # Image
        img = Image.open(path)
        img = self.transform(img.convert('RGB'))
        # Annotations
        anns = torch.zeros((len(ann_ids), 5), dtype=torch.float)
        # [[x, y, w, h, category]
        #           ...
        #  [x, y, w, h, category]]
        for idx, annId in enumerate(ann_ids):
            ann = self.cocotool.anns[annId]
            anns[idx, :4] = torch.tensor(ann['bbox'])
            anns[idx, 4] = torch.tensor(ann['category_id'])
        # Extracted regions
        extracted_regions = None
        if self.ss_roi:
            cv_img = cv2.imread(path)
            cv_img = cv2.resize(cv_img, dsize=(self.input_size))
            extracted_regions = selective_search_extraction(cv_img, search_type='fast')
            extracted_regions = torch.from_numpy(extracted_regions).float()

        return img, anns, extracted_regions

    def __len__(self):
        return len(self.imgIds)

    def _rescale_bboxes(self, input_size):
        # rescaling bounding boxes to corresponding input shape
        for imgId in self.imgIds:
            ann_ids = self.cocotool.getAnnIds(imgId)

            img_width = self.cocotool.imgs[imgId]['width']
            img_height = self.cocotool.imgs[imgId]['height']

            img_size = (img_width, img_height)

            for annId in ann_ids:
                self.cocotool.anns[annId]['bbox'] = bbox_resize(self.cocotool.anns[annId]['bbox'], img_size, input_size)

    def _init_cocotools(self, data_dir, data_type, data_size, train):

        ann_types = ['segm', 'bbox', 'keypoints']
        ann_type = ann_types[1]  # specify type here
        prefix = 'person_keypoints' if ann_type == 'keypoints' else 'instances'
        ann_file = f'{data_dir}/annotations/{prefix}_{data_type}2017.json'  # 标注文件

        cocotool = COCO(ann_file)

        img_ids = sorted(cocotool.getImgIds())
        if not data_size == 'full':
            img_ids = img_ids[0:data_size]
        if not train:
            img_ids = img_ids[-data_size:]

        return cocotool, img_ids

    def _target_normalize_coef(self, eps=1e-8):

        cls_counts = np.zeros((len(self.category) + 1, 1)) + eps
        sums = np.zeros((len(self.category) + 1, 4))
        squared_sums = np.zeros((len(self.category) + 1, 4))

        for imgId in self.imgIds:
            ann_ids = self.cocotool.getAnnIds(imgId)
            for ann_id in ann_ids:
                ann = self.cocotool.anns[ann_id]
                category = ann['category_id']
                bbox = ann['bbox']
                cls_counts[self.numbered_category.index(category)] += 1
                sums[self.numbered_category.index(category), :] += bbox
                squared_sums[self.numbered_category.index(category), :] += bbox ** 2

        means = sums / cls_counts
        stds = np.sqrt(squared_sums / cls_counts - means ** 2)

        return means, stds


def coco_collate(batch):
    imgs = torch.stack([item[0] for item in batch], 0)
    anns = torch.stack([item[1] for item in batch], 0)
    regions = torch.stack([item[2] for item in batch], 0)
    return imgs, anns, regions


if __name__ == '__main__':
    cocodb = COCODataset(data_dir="D:/dataset/COCO2017")

    coco_loader = torch.utils.data.DataLoader(cocodb)

    a, a_anns, reg = cocodb[0]

    print("a_anns shape : ", a_anns.shape)
    print("a shape : ", a.shape)
    print("cocodb target means shape : ", cocodb.target_means.shape)
    print("cocodb target stds shape : ", cocodb.target_stds.shape)
    print("region proposals shape : ", reg.shape)
