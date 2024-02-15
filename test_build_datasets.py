import argparse

import opts
from datasets.ytvos_online import YTVOSDataset

from pathlib import Path
import util.misc as utils
import torch
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset, DataLoader
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

def get_datasets_task(metas,task_id):
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
    # args.__setattr__('num_frames',5)

    dataset_train = YTVOSDataset(image_set, img_folder, ann_file,
                           transforms=make_coco_transforms(image_set, max_size=640),
                           return_masks=True,
                           num_frames=2, max_skip=4, sampler_interval=4,
                           args=args)

    dataset_train.update_metas(metas,task_id)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)


    return dataset_train, data_loader_train
def get_task_metas(task_id):
    task_metas = []
    ROOT = '/home/user/OnlineRefer_Audio_CIL/data/'
    with open(ROOT + 'metas.json', 'r') as f:
        # metas is the list
        metas = json.load(f)['metas']
    with open(ROOT + 'task{}.json'.format(task_id), 'r') as f:
        tasks = json.load(f)[str(task_id)]

    numbers = []
    for category in tasks:
        number = 0
        exps = tasks[category]
        for exp in exps:
            number += 1
            ids = tasks[category][exp]
            for id in ids:
                task_metas.append(metas[id])
        numbers.append(number)

    return metas,task_metas,list(tasks.keys()),numbers




if __name__ == '__main__':
    task_id = 0
    metas,task_metas,categories,numbers = get_task_metas(task_id)
    print(len(metas),len(task_metas),categories,numbers)

    dataset_train, data_loader_train =get_datasets_task(task_metas,task_id)
    print(len(dataset_train),len(data_loader_train))

    for i in range(10):
        dataset_train.__getitem__(i)
    for data in dataset_train.metas:
        print(data)


    # for sample,targets  in data_loader_train:
    #     print(targets)
    #
    #

    # print(dataset.metas[1])
    # print(dataset.metas[2])
    # import json
    # # with open('/home/user/OnlineRefer_Modified/data/Datasets/ref-youtube-vos/meta_expressions/train/sign_meta_expressions.json','r') as f:
    # #     data = json.load(f)
    # #
    # # print(data)
    # for task_meta in task_metas:
    #     print(task_meta)
