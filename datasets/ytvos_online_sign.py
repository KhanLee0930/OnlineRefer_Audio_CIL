"""
Ref-YoutubeVOS data loader
"""
from pathlib import Path

import torch
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
from transformers import VivitImageProcessor, AutoImageProcessor

import datasets.transforms_video as T

import os
from PIL import Image
import json
import numpy as np
import random

from datasets.categories import ytvos_category_dict as category_dict


class YTVOSDataset(Dataset):
    """
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the first
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.

    """

    def __init__(self, image_set,img_folder: Path, ann_file: Path, transforms, return_masks: bool,
                 num_frames: int, max_skip: int, sampler_interval: int, args=None):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.return_masks = return_masks  # not used
        self.num_frames = num_frames
        self.num_clips = args.num_clips
        self.max_skip = max_skip
        self.sampler_interval = sampler_interval
        # self.sampler_interval = 3
        self.reverse_aug = False
        # create video meta data
        self.aud_file = '/home/user/OnlineRefer_Modified/data/Datasets/ref-youtube-vos-audio-encoded/'+image_set
        self.sign_file = '/home/user/OnlineRefer_Modified/data/Datasets/ref-youtube-vos-sign/' + image_set
        self.image_processor = AutoImageProcessor.from_pretrained("/home/user/OnlineRefer_Modified/videoMAE/",local_files_only=True)

        self.prepare_metas()
        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))
        print('\n')

        # video sampler.
        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths

        self.current_epoch = 0
        print("sampler_steps={} lengths={}".format(self.sampler_steps, self.lengths))

    def prepare_metas(self):
        # read object information
        with open(os.path.join(str(self.img_folder), 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']

        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in self.videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            for exp_id, exp_dict in vid_data['expressions'].items():
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp_dict['exp']
                    meta['obj_id'] = int(exp_dict['obj_id'])
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    # get object category
                    obj_id = exp_dict['obj_id']
                    meta['audio'] = exp_id
                    meta['category'] = vid_meta['objects'][obj_id]['category']
                    self.metas.append(meta)

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.current_epoch = self.current_epoch + 1

        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        self.period_idx = 0
        for i in range(len(self.sampler_steps)):
            if self.current_epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(self.current_epoch, self.period_idx))
        self.num_frames = self.lengths[self.period_idx]

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax  # y1, y2, x1, x2

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]  # dict

            video, exp, obj_id, category, frames, frame_id, audio_id = \
                meta['video'], meta['exp'], meta['obj_id'], meta['category'], meta['frames'], meta['frame_id'],meta['audio']
            # clean up the caption
            exp = " ".join(exp.lower().split())
            category_id = category_dict[category]
            vid_len = len(frames)

            num_frames = self.num_frames * self.num_clips
            # random sparse sample
            sample_indx = [frame_id]
            if num_frames != 1:
                # local sample
                sample_id_before = random.randint(1, self.sampler_interval)
                sample_id_after = random.randint(1, self.sampler_interval)
                local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                sample_indx.extend(local_indx)

                # maximum number of input frames is 5 for online mode
                if num_frames > 3:
                    sample_id_before = random.randint(1, self.sampler_interval)
                    sample_id_after = random.randint(1, self.sampler_interval)
                    local_indx = [max(0, frame_id - self.sampler_interval - sample_id_before),
                                  min(vid_len - 1, frame_id + self.sampler_interval + sample_id_after)]
                    sample_indx.extend(local_indx)

            sample_indx.sort()
            if random.random() < self.reverse_aug:
                sample_indx = sample_indx[::-1]

            # read frames and masks
            imgs, labels, boxes, masks, valid = [], [], [], [], []
            for j in range(num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('P')

                # create the target
                label = torch.tensor(category_id)
                mask = np.array(mask)
                mask = (mask == obj_id).astype(np.float32)  # 0,1 binary
                if (mask > 0).any():
                    y1, y2, x1, x2 = self.bounding_box(mask)
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valid.append(1)
                else:  # some frame didn't contain the instance
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                    valid.append(0)
                mask = torch.from_numpy(mask)

                # append
                imgs.append(img)
                labels.append(label)
                masks.append(mask)
                boxes.append(box)

            # transform
            w, h = img.size
            labels = torch.stack(labels, dim=0)
            boxes = torch.stack(boxes, dim=0)
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            masks = torch.stack(masks, dim=0)
            target = {
                'frames_idx': torch.tensor(sample_indx),  # [T,]
                'labels': labels,  # [T,]
                'boxes': boxes,  # [T, 4], xyxy
                'masks': masks,  # [T, H, W]
                'valid': torch.tensor(valid),  # [T,]
                'caption': exp,
                'orig_size': torch.as_tensor([int(h), int(w)]),
                'size': torch.as_tensor([int(h), int(w)]),
                'audio': torch.load(self.aud_file + '/' + video + '/' + audio_id + '.pt'),
                'sign':self.video_pixel_values(self.sign_file + '/' + video + '/' + audio_id + '/' + 'video.mp4')
            }

            # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
            imgs, target = self._transforms(imgs, target)
            imgs = torch.stack(imgs, dim=0)  # [T, 3, H, W]

            # FIXME: handle "valid", since some box may be removed due to random crop
            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return imgs, target

    import av
    def video_pixel_values(self,file_path):
        import av
        container = av.open(file_path)
        # sample 32 frames
        indices = sample_frame_indices(clip_len=16, frame_sample_rate=container.streams.video[0].frames // 16,
                                       seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices)
        inputs = self.image_processor(list(video), return_tensors="pt")
        return inputs['pixel_values']

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


def build(image_set, args):
    root = Path(args.ytvos_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "meta_expressions" / "train" / "sign_meta_expressions.json"),
        "val": (root / "valid", root / "meta_expressions" / "val" / "meta_expressions.json"),  # not used actually
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = YTVOSDataset(image_set,img_folder, ann_file, transforms=make_coco_transforms(image_set, max_size=args.max_size),
                           return_masks=args.masks,
                           num_frames=args.num_frames, max_skip=args.max_skip, sampler_interval=args.sampler_interval,
                           args=args,)
    return dataset

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices