import math
import copy
import random
import functools

import torch
import numpy as np
from torch.utils.data import Dataset

from dsets import Ct, get_candidate_info_list
from utils import xyz2irc, get_cache, logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

raw_cache = get_cache('raw')

@functools.lru_cache(1)
def get_candidate_info_dict(require_on_disk_bool=True):
    # Returns a dictionary with the series uid as key and list with each candidate nodule
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

        # List will all candidates of this specific CT scan instance
        candidate_info_list = get_candidate_info_dict()[self.series_uid]

        # All candidates that actually are a nodule
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
        # Create a 3D bounding box mask around the center of an actual nodule
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
        # Return the ct chunk
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


@raw_cache.memoize(typed=True)
def get_ct_sample_size(series_uid):
    ct = SegmentationCt(series_uid)
    return int(ct.hu_a.shape[0]), ct.positive_indexes


class Luna2dSegmentationDataset(Dataset):
    def __init__(
            self, data_split='train', ratios=(0.8, 0.1, 0.1),
            series_uid=None, context_slice_count=3, full_ct=False, require_on_disk=True
    ):
        r_train, r_eval, r_test = ratios
        self.context_slice_count = context_slice_count
        self.full_ct = full_ct

        # Only contains series_uids
        if series_uid:
            temp_series_list = [series_uid]
        else:
            temp_series_list = sorted(get_candidate_info_dict(require_on_disk).keys())

        if series_uid is None:
            sample_count = len(temp_series_list)

            n_train = math.floor(sample_count * r_train)
            n_eval = math.floor(sample_count * r_eval)

            s_train = temp_series_list[:n_train]
            s_eval = temp_series_list[n_train:n_train + n_eval]
            s_test = temp_series_list[n_train+n_eval:]

            if data_split == 'train':
                self.series_list = s_train
            elif data_split == 'eval':
                self.series_list = s_eval
            elif data_split == 'test':
                self.series_list = s_test

            assert self.series_list, f'Dataset is empty'
        else:
            self.series_list = temp_series_list

        if data_split == 'eval':
            self.series_list = []
            for i in range(len(temp_series_list)):
                if i % 5 == 0 and i % 10 != 0:
                    self.series_list.append(temp_series_list[i])
            assert self.series_list
        elif data_split == 'test':
            self.series_list = []
            for i in range(len(temp_series_list)):
                if i % 5 != 0 and i % 10 == 0:
                    self.series_list.append(temp_series_list[i])
            assert self.series_list
        elif data_split == 'train':
            self.series_list = []
            for i in range(len(temp_series_list)):
                if i % 5 != 0 and i % 10 != 0:
                    self.series_list.append(temp_series_list[i])
            assert self.series_list
        else:
            self.series_list = temp_series_list

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

        self.candidate_info_list = get_candidate_info_list(require_on_disk)
        series_set = set(self.series_list)

        # Filter for series UIDs in our series list
        self.candidate_info_list = [
            cit for cit in self.candidate_info_list if cit.series_uid in series_set
        ]

        # List of samples with nodule present
        self.pos_list = [
            nt for nt in self.candidate_info_list if nt.isNodule_bool
        ]

        log.info('{}: {} {} series, {} slices, {} nodules'.format(
            self,
            len(self.series_list),
            data_split,
            len(self.sample_list),
            len(self.pos_list)
        ))

    def __len__(self):
        return len(self.sample_list)

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

    def epoch_reset(self, update_ratio):
        self.shuffle_samples()
        if self.ratio_int and update_ratio:
            self.ratio_int += 1

    def shuffle_samples(self):
        random.shuffle(self.candidate_info_list)
        random.shuffle(self.pos_list)

    def __len__(self):
        return 300_000

    def __getitem__(self, index):
        candidate_info_tup = self.pos_list[index % len(self.pos_list)]
        return self.getitem_training_crop(candidate_info_tup)

    @staticmethod
    def getitem_training_crop(candidate_info_tup):
        # Creates a random 64 x 64 crop of the 96 x 96 area around the nodule center
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


class PrecacheLunaDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.candidate_info_list = copy.copy(get_candidate_info_list(kwargs['require_on_disk']))
        self.pos_list = [nt for nt in self.candidate_info_list if nt.isNodule_bool]

        self.seen_set = set()
        self.candidate_info_list.sort(key=lambda x: x.series_uid)

    def __len__(self):
        return len(self.candidate_info_list)

    def __getitem__(self, ndx):
        candidate_info_tup = self.candidate_info_list[ndx]
        get_ct_raw_candidate_with_pos_mask(candidate_info_tup.series_uid, candidate_info_tup.center_xyz, (7, 96, 96))

        series_uid = candidate_info_tup.series_uid
        if series_uid not in self.seen_set:
            self.seen_set.add(series_uid)

            get_ct_sample_size(series_uid)

        return 0, 1
