import argparse

import opts
from datasets.ytvos_online import YTVOSDataset

from pathlib import Path

import torch
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
# from transformers import VivitImageProcessor, AutoImageProcessor

import datasets.transforms_video as T

import os
from PIL import Image
import json
import numpy as np
import random

from datasets.categories import ytvos_category_dict as category_dict


def make_coco_transforms(image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ])
            ),
            normalize,
        ])

    # we do not use the 'val' set since the annotations are inaccessible
    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            # T.RandomResize([240], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def devide_category(number, categories):
    category_name = list(categories.keys())

    random.shuffle(category_name)

    group_size = len(category_name) // number
    groups = [category_name[i:i + group_size] for i in range(0, len(category_name), group_size)]
    return groups

ytvos_path = '/home/user/OnlineRefer_Modified/data/Datasets/ref-youtube-vos'
image_set = 'train'
root = Path(ytvos_path)
assert root.exists(), f'provided YTVOS path {root} does not exist'
PATHS = {
    "train": (root / "train", root / "meta_expressions" / "train" / "meta_expressions.json"),
    "val": (root / "valid", root / "meta_expressions" / "val" / "meta_expressions.json"),  # not used actually
}
img_folder, ann_file = PATHS[image_set]

parser = argparse.ArgumentParser('RVOSNet training and evaluation script', parents=[opts.get_args_parser()])
args = parser.parse_args()

dataset = YTVOSDataset(image_set, img_folder, ann_file,
                       transforms=make_coco_transforms(image_set, max_size=640),
                       return_masks=True,
                       num_frames=2, max_skip=4, sampler_interval=4,
                       args=args)

# categories = {}
# cil_dataset = {}
# cil_dataset['metas'] = dataset.metas
# cil_dataset['tasks'] = {}
#
# for idx, meta in enumerate(dataset.metas):
#     if meta['category'] not in categories:
#         categories[meta['category']] = {}
#     if meta['exp'] not in categories[meta['category']]:
#         categories[meta['category']][meta['exp']] = []
#
#     categories[meta['category']][meta['exp']].append(idx)
#
#
# for task_id,group in enumerate(devide_category(5,categories)):
#     cil_dataset['tasks'][str(task_id)] = {}
#     for category in group:
#         category_data = {}
#         exps = list(categories[category].keys())
#         samples = random.sample(exps, 30)
#         for sample in samples:
#             category_data[sample] = categories[category][sample]
#         cil_dataset['tasks'][str(task_id)][category] = category_data
#
# with open('/home/user/OnlineRefer_Modified/data/CIL_Datasets/cil_expressions.json','w') as f:
#     json.dump(cil_dataset,f)

with open('/home/user/OnlineRefer_Modified/data/CIL_Datasets/cil_expressions.json','r') as f:
    cli_dataset = json.load(f)
print(cli_dataset['tasks']['1'])