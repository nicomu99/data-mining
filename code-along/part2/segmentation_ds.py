import random
import functools

import torch
import numpy as np
from torch.utils.data import Dataset

from dsets import Ct, get_candidate_info_list
from utils import xyz2irc, get_cache

raw_cache = get_cache('part2ch13_raw')

@functools.lru_cache(1)
def get_candidate_info_dict(require_on_disk_bool=True):
    candidate_info_list = get_candidate_info_list(require_on_disk_bool)

    candidate_info_dict = {}
    for candidate_info_tup in candidate_info_list:
        candidate_info_dict.setdefault(
            candidate_info_tup.series_uid, []
        ).append(candidate_info_tup)

    return candidate_info_dict

class SegmentationCt(Ct):
    def __init__(self, series_uid):
        super().__init__(series_uid, clip=False)

        candidate_info_list = get_candidate_info_dict()[self.series_uid]

        self.positive_info_list = [
            candidate_tup
            for candidate_tup in candidate_info_list
            if candidate_tup.isNodule_bool
        ]
        self.positive_mask = self.build_annotation_mask(self.positive_info_list)

        # First get 1D vector with number of voxels flagged as nodule in each slice
        # Then choose indices that have at least 1 positive flag
        self.positive_indexes = self.positive_mask.sum(axis=(1, 2)).nonzero()[0].tolist()

    def build_annotation_mask(self, positive_info_list, threshold_hu = -700):
        bounding_box_a = np.zeros_like(self.hu_a, dtype=np.bool)

        for candidate_info_tup in positive_info_list:
            center_irc = xyz2irc(
                candidate_info_tup.center_xyz,
                self.origin_xyz,
                self.vx_size_xyz,
                self.direction_a
            )
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2
            try:
                while self.hu_a[ci + index_radius, cr, cc] > threshold_hu and \
                    self.hu_a[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                while self.hu_a[ci, cr + row_radius, cc] > threshold_hu and \
                    self.hu_a[ci, ci - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while self.hu_a[ci, cr, cc + col_radius] > threshold_hu and \
                        self.hu_a[ci, ci, cc + col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            bounding_box_a[
                ci - index_radius: ci + index_radius + 1,
                cr - row_radius: cr + row_radius + 1,
                cc - col_radius: cc + col_radius + 1
            ] = True

        mask_a = bounding_box_a & (self.hu_a > threshold_hu)
        return mask_a

    def get_raw_candidate(self, center_xyz, width_irc):
        ct_chunk, center_irc = super().get_raw_candidate(center_xyz, width_irc)
        pos_chunk = self.positive_mask[tuple(self.slice_list)]
        return ct_chunk, pos_chunk, center_irc

@functools.lru_cache(1, typed=True)
def get_segmentation_ct(series_uid):
    return SegmentationCt(series_uid)

@raw_cache.memoize(typed=True)
def get_ct_raw_candidate_with_pos_mask(series_uid, center_xyz, width_irc):
    ct = get_segmentation_ct(series_uid)
    ct_chunk, pos_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
    ct_chunk.clip(-1000, 1000, ct_chunk)
    return ct_chunk, pos_chunk, center_irc

@raw_cache.memoize(type=True)
def get_ct_sample_size(series_uid):
    ct = SegmentationCt(series_uid)
    return int(ct.hu_a.shape[0]), ct.positive_indexes

class Luna2dSegmentationDataset(Dataset):
    def __init__(self, val_stride=0, is_val_set=False, series_uid=None, context_slice_count=3, full_ct=False):
        self.context_slice_count = context_slice_count
        self.full_ct = full_ct

        # Only contains series_uids
        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(get_candidate_info_dict().keys())

        if is_val_set:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        # Contains a tuple of (series_uid, slice_index) of candidates to check
        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indexes = get_ct_sample_size(series_uid)

            if self.full_ct:
                # Pick full CT, including slices where no nodule mask is present
                self.sample_list += [
                    (series_uid, slice_index)
                    for slice_index in range(index_count)
                ]
            else:
                # Only pick slices where a nodule mask is present
                self.sample_list += [
                    (series_uid, slice_index)
                    for slice_index in positive_indexes
                ]

        self.candidate_info_list = get_candidate_info_list()
        series_set = set(self.series_list)

        # Filter for series UIDs in our series list
        self.candidate_info_list = [
            cit for cit in self.candidate_info_list if cit.series_uid in series_set
        ]

        # List of samples with nodule present
        self.pos_list = [
            nt for nt in self.candidate_info_list if nt.isNodule_bool
        ]

    def __getitem__(self, index):
        series_uid, slice_index = self.sample_list[index % len(self.sample_list)]   # Wrap if we run out of samples
        return self.getitem_full_slice(series_uid, slice_index)

    def getitem_full_slice(self, series_uid, slice_index):
        ct = get_segmentation_ct(series_uid)
        ct_t = torch.zeros((self.context_slice_count * 2 + 1, 512, 512))

        start_index = slice_index - self.context_slice_count
        end_index = slice_index + self.context_slice_count + 1
        for i, context_index in enumerate(range(start_index, end_index)):
            # Duplicate slices if we reach bounds
            context_index = max(context_index, 0)
            context_index = min(context_index, ct.hu_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_a[context_index].astype(np.float32))
        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(ct.positive_mask[slice_index]).unsqueeze(0)

        return ct_t, pos_t, ct.series_uid, slice_index

class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ratio_int = 2

    def __len__(self):
        return 300_000

    def __getitem__(self, index):
        candidate_info_tup = self.pos_list[index % len(self.pos_list)]
        return self.getitem_training_crop(candidate_info_tup)

    def getitem_training_crop(self, candidate_info_tup):
        ct_a, pos_a, center_irc = get_ct_raw_candidate_with_pos_mask(
            candidate_info_tup.series_uid,
            candidate_info_tup.center_xyz,
            (7, 96, 96)
        )
        pos_a = pos_a[3:4]

        # Extract random crop of shape 32x32
        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)
        ct_t = torch.from_numpy(
            ct_a[:, row_offset:row_offset + 64, col_offset:col_offset + 64]
        ).to(torch.float32)
        pos_t = torch.from_numpy(
            pos_a[:, row_offset:row_offset + 64, col_offset:col_offset + 64]
        ).to(torch.long)

        slice_index = center_irc.index
        return ct_t, pos_t, candidate_info_tup.series_uid, slice_index
