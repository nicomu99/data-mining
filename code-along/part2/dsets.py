import os
import glob
import csv
import functools
import copy
import random
import math

from collections import namedtuple

import numpy as np
import SimpleITK as Sitk

from utils import get_cache
from utils import XyzTuple, xyz2irc
from utils import logging
from utils import get_data_root

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

raw_cache = get_cache('raw')

BASE_PATH = get_data_root()

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz'
)

@functools.lru_cache(1)     # Caches the most recent call with same argument, i.e. the return value is not recomputed
def get_candidate_info_list(require_on_disk=True, reverse=True):
    # Mhd is a header file for image metadata
    mhd_list = glob.glob(os.path.join(BASE_PATH, 'data/subset*/*.mhd'))     # Get all files with ending .mhd
    present_on_disk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}     # Extracts the seriesuid from path name

    # Retrieve center and diameter from annotations.csv
    candidate_info_list = []
    with open('data/annotations_with_malignancy.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:     # Skip header row
            series_uid = row[0]
            annotation_center_xyz = tuple([float(x) for x in row[1:4]])
            annotation_diameter_mm = float(row[4])
            is_mal = False if row[5] == 'False' else True

            if series_uid not in present_on_disk_set and require_on_disk:
                continue

            candidate_info_list.append(
                CandidateInfoTuple(
                    True, True, is_mal, annotation_diameter_mm, series_uid, annotation_center_xyz
                )
            )

    with open('data/candidates.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:     # Again skip header
            series_uid = row[0]

            if series_uid not in present_on_disk_set and require_on_disk:
                # Series uid is in a subset not present on disk
                continue

            is_nodule_bool = bool(int(row[4]))
            candidate_center_xyz = tuple([float(x) for x in row[1:4]])

            if not is_nodule_bool:
                candidate_info_list.append(
                    CandidateInfoTuple(
                        False, False, False, 0.0, series_uid, candidate_center_xyz
                    )
                )

    # Sort is according to CandidateInfoTuple content order:
    #   Is nodule comes first (True > 0)
    #   Then diameter size
    #   ...
    candidate_info_list.sort(reverse=reverse)
    return candidate_info_list

class Ct:
    # Class for holding each individual ct scan
    def __init__(self, series_uid, clip=True):
        mhd_path = glob.glob(os.path.join(BASE_PATH, f'data/subset*/{series_uid}.mhd'))[0]

        ct_mhd = Sitk.ReadImage(mhd_path)       # Implicitly consumes the .raw file in addition to .mhd
        ct_a = np.array(Sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)   # Shape (slice, height, width)
        if clip:
            ct_a.clip(-1000, 1000, ct_a)        # All values below -1000 get bounded to -1000, all above to 1000

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vx_size_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)
        self.slice_list = None

    def get_raw_candidate(self, center_xyz, width_irc):
        # Takes the center expressed in xyz and voxel width
        # Returns cubic chunk of CT and center as IRC coordinates of a candidate nodule
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vx_size_xyz, self.direction_a)

        self.slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert 0 <= center_val < self.hu_a.shape[axis], repr(
                [self.series_uid, self.origin_xyz, self.vx_size_xyz, center_irc, axis]  # Used for error message
            )

            # Make sure indexes are in the correct bounds
            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            self.slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(self.slice_list)]     # Contains the chunk with candidate nodule
        return ct_chunk, center_irc

# Cache the most recent ct in memory
# Several candidates have the same ct, so it does not have to be reloaded to memory every time
# But order matters!
@functools.lru_cache(1, typed=True)
def get_ct(series_uid):
    return Ct(series_uid)

# The calls to get_ct_raw_candidate are cached on the disk, i.e. we cache the float arrays of the candidate areas
# This significantly reduces run time, as the size of what is loaded is decrease from 2^25 to 2^15 elements per ct
# Purpose: Cache the results of an expensive CT scan preprocessing to avoid re-computation.
@raw_cache.memoize(typed=True)
def get_ct_raw_candidate(series_uid, center_xyz, width_irc):
    ct = get_ct(series_uid)
    ct_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
    return ct_chunk, center_irc

def load_chunk(series_uid, center_xyz, width_irc, use_cache=True):
    if use_cache:
        ct_chunk, center_irc = get_ct_raw_candidate(series_uid, center_xyz, width_irc)
    else:
        ct = get_ct(series_uid)
        ct_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
    return torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32), center_irc   # Add batch and channel dim

def get_ct_augmented_candidate(
        augmentation_dict, series_uid, center_xyz, width_irc,
        mixup_series_uid, mixup_center_xyz, use_cache=True
):
    ct_t, center_irc = load_chunk(series_uid, center_xyz, width_irc, use_cache)

    if 'mixup' in augmentation_dict and mixup_series_uid is not None:
        mixup_t, _ = load_chunk(mixup_series_uid, mixup_center_xyz, width_irc, use_cache)

        lambda_val = random.betavariate(0.4, 0.4)
        ct_t = lambda_val * ct_t + (1 - lambda_val) * mixup_t


    transform_t = torch.eye(4)
    for i in range(3):
        if 'flip' in augmentation_dict:
            # Flip each dimension randomly
            if random.random() > 0.5:
                transform_t[i, i] *= -1     # If t[i, i] = -1 -> mirror

        if 'offset' in augmentation_dict:
            # Add offset: Makes the model more robust to off-center nodules
            offset_float = augmentation_dict['offset']  # Controls the impact in %: E.g. 0.2 -> 20% of image size
            random_float = (random.random() * 2 - 1)    # Random float between -1 and 1
            transform_t[i, 3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            # Add scaling: Zoom in or out
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)    # Again -1 and 1: If > 0 zoom in
            transform_t *= 1.0 + scale_float * random_float

    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2       # Random angle between 0 and 360
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)
        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        transform_t @= rotation_t   # Apply the transformation

    # Construct a map of coordinates that carries out the transformation of the ct scan
    affine_t = F.affine_grid(
        # Affine gird expects 3x4 for 3D images
        # [:3]: Pick first three rows, shape (3, 4)
        # .unsqueeze(0): add num samples dimension (in this case 1), shape (1, 3, 4)
        transform_t[:3].unsqueeze(0).to(torch.float32),
        list(ct_t.size()),
        align_corners=False,
    )

    # Carry out the transformation by pulling voxel values from the original ct scan and re-placing them
    augmented_chunk = F.grid_sample(
        ct_t,
        affine_t,               # Specifies the sampling pixel location normalized by input spatial dims
        padding_mode='border',  # Border values are used for out of bounds grid locations
        align_corners=False
    ).to('cpu')

    if 'noise' in augmentation_dict:
        # Add random noise
        # Random tensor of same shape as input and random values between 0 and 1 from std. normal dist.
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    def __init__(
            self,
            val_stride = 0,
            is_val_set = None,
            series_uid = None,
            ratio_int = None,
            augmentation_dict = None,
            candidate_info_list = None,
            reverse = True,
            require_on_disk = True
    ):
        # Controls the ratio between positive and negative samples
        # E.g. ratio_int = 2: 2 negative samples, 1 positive
        # Also, if used, the dataset size is capped at 200_000 samples
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict
        self.require_on_disk = require_on_disk

        if candidate_info_list:
            self.candidate_info_list = copy.copy(candidate_info_list)
            self.use_cache = False
        else:
            self.candidate_info_list = copy.copy(get_candidate_info_list(self.require_on_disk, reverse=reverse))
            self.use_cache = True

        # If series uid is passed, we only get candidates from that scan
        if series_uid:
            self.candidate_info_list = [
                x for x in self.candidate_info_list if x.series_uid == series_uid
            ]

        if is_val_set:
            # If is validation set, retrieve every val_stride-th element
            assert val_stride > 0, val_stride
            self.candidate_info_list = self.candidate_info_list[::val_stride]
            assert self.candidate_info_list
        elif val_stride > 0:
            # Else delete every val_stride-th element
            del self.candidate_info_list[::val_stride]
            assert self.candidate_info_list

        # Save true and false samples in separate lists to control the ratio between them during training
        self.negative_list = [
            nt for nt in self.candidate_info_list if not nt.isNodule_bool
        ]
        self.positive_list = [
            pt for pt in self.candidate_info_list if pt.isNodule_bool
        ]

        log.info("{!r}: {} {} samples {}".format(
            self,
            len(self.candidate_info_list),
            "validation" if is_val_set else "training",
            require_on_disk
        ))

    def epoch_reset(self, update_ratio):
        self.shuffle_samples()
        if self.ratio_int and update_ratio:
            self.ratio_int += 1

    def shuffle_samples(self):
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.positive_list)

    def __len__(self):
        if self.ratio_int:
            return 200_000
        else:
            return len(self.candidate_info_list)

    def __getitem__(self, index):
        """
        Returns item at index `index`.
        :param index: The index of the candidate we want to fetch.
        :return: Tuple with candidate chunk, one-hot encoded label, series uid and center of chunk.
        """
        mixup_info_tup = None
        if self.ratio_int:
            pos_index = index // (self.ratio_int + 1)

            # E.g. if ratio int = 2: Every third index will be positive
            if index % (self.ratio_int + 1):
                neg_index = index - 1 - pos_index
                neg_index %= len(self.negative_list)    # If index > number of negative samples -> Restart from top
                candidate_info_tup = self.negative_list[neg_index]
            else:
                pos_index %= len(self.positive_list)    # We run out of positive samples before all iterations finished
                candidate_info_tup = self.positive_list[pos_index]

                # For mixup data augmentation
                random_pos_sample_index = random.randint(0, len(self.positive_list) - 1)
                mixup_info_tup = self.positive_list[random_pos_sample_index]
        else:
            candidate_info_tup = self.candidate_info_list[index]
        width_irc = (32, 48, 48)

        if self.augmentation_dict:
            candidate_t, center_irc = get_ct_augmented_candidate(
                self.augmentation_dict,
                candidate_info_tup.series_uid,
                candidate_info_tup.center_xyz,
                width_irc,
                mixup_info_tup.series_uid if mixup_info_tup is not None else None,
                mixup_info_tup.center_xyz if mixup_info_tup is not None else None,
                self.use_cache,
            )
        elif self.use_cache:
            candidate_a, center_irc = get_ct_raw_candidate(
                candidate_info_tup.series_uid,
                candidate_info_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            ct = get_ct(candidate_info_tup.series_uid)
            candidate_a, center_irc = ct.get_raw_candidate(
                candidate_info_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        # Classification tensor for CrossEntropy
        pos_t = torch.tensor([
                not candidate_info_tup.isNodule_bool,
                candidate_info_tup.isNodule_bool
            ],
            dtype=torch.long
        )

        return (
            candidate_t,
            pos_t,
            candidate_info_tup.series_uid,
            torch.tensor(center_irc)
        )