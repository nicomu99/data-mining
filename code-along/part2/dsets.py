import os
import glob
import csv
import functools
import copy

from collections import namedtuple

import numpy as np
import SimpleITK as Sitk

from utils import get_cache
from utils import XyzTuple, xyz2irc
from utils import logging
from utils import get_data_root

import torch
import torch.cuda
from torch.utils.data import Dataset

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

raw_cache = get_cache('part2ch10_raw')

BASE_PATH = get_data_root()

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz'
)

@functools.lru_cache(1)     # Caches the most recent call with same argument, i.e. the return value is not recomputed
def get_candidate_info_list(require_on_disk=True):
    # Mhd is a header file for image metadata
    mhd_list = glob.glob(os.path.join(BASE_PATH, 'data/subset*/*.mhd'))      # Get all files with ending .mhd
    present_on_disk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}     # Extracts the seriesuid from path name

    # Retrieve center and diameter from annotations.csv
    diameter_dict = {}
    annotation_file = os.path.join(BASE_PATH, 'data/annotations.csv')
    with open(annotation_file, 'r') as f:
        for row in list(csv.reader(f))[1:]:     # Skip header row
            series_uid = row[0]
            annotation_center_xyz = tuple([float(x) for x in row[1:4]])
            annotation_diameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
                (annotation_center_xyz, annotation_diameter_mm)
            )

    candidate_info_list = []
    candidate_file = os.path.join(BASE_PATH, 'data/candidates.csv')
    with open(candidate_file, 'r') as f:
        for row in list(csv.reader(f))[1:]:     # Again skip header
            series_uid = row[0]

            if series_uid not in present_on_disk_set and require_on_disk:
                # Series uid is in a subset not present on disk
                continue

            is_nodule_bool = bool(int(row[4]))
            candidate_center_xyz = tuple([float(x) for x in row[1:4]])

            # Diameter information in annotations.csv and candidates.csv is not uniform
            candidate_diameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotation_center_xyz, annotation_diameter_mm = annotation_tup
                for i in range(3):
                    # Make sure candidate centers are not too far apart between the two files
                    delta_mm = abs(candidate_center_xyz[i] - annotation_center_xyz[i])
                    if delta_mm > annotation_diameter_mm / 4:   # / 2 to get radius and again by / 2 to limit distance
                        break
                else:   # Executes only if for completes without hitting break
                    # If centers are too far apart, the candidate has diameter 0.0
                    candidate_diameter_mm = annotation_diameter_mm
                    break

            candidate_info_list.append(
                CandidateInfoTuple(
                    is_nodule_bool, candidate_diameter_mm, series_uid, candidate_center_xyz
                )
            )

    # Sort is according to CandidateInfoTuple content order:
    #   Is nodule comes first (True > 0)
    #   Then diameter size
    #   ...
    candidate_info_list.sort(reverse=True)
    return candidate_info_list

class Ct:
    # Class for holding each individual ct scan
    def __init__(self, series_uid):
        mhd_path = glob.glob(os.path.join(BASE_PATH, f'data/subset*/{series_uid}.mhd'))[0]

        ct_mhd = Sitk.ReadImage(mhd_path)       # Implicitly consumes the .raw file in addition to .mhd
        ct_a = np.array(Sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)   # Shape (slice, height, width)
        ct_a.clip(-1000, 1000, ct_a)        # All values below -1000 get bounded to -1000, all above to 1000

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vx_size_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def get_raw_candidate(self, center_xyz, width_irc):
        # Takes the center expressed in xyz and voxel width
        # Returns cubic chunk of CT and center as IRC coordinates of a candidate nodule
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vx_size_xyz, self.direction_a)

        slice_list = []
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

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]     # Contains the chunk with candidate nodule
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


class LunaDataset(Dataset):
    def __init__(self, val_stride = 0, is_val_set = None, series_uid = None, require_on_disk = True):
        self.candidate_info_list = copy.copy(get_candidate_info_list(require_on_disk))

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

        log.info("{!r}: {} {} samples {}".format(
            self,
            len(self.candidate_info_list),
            "validation" if is_val_set else "training",
            require_on_disk
        ))

    def __len__(self):
        return len(self.candidate_info_list)

    def __getitem__(self, index):
        """
        Returns item at index `index`.
        :param index: The index of the candidate we want to fetch.
        :return: Tuple with candidate chunk, one-hot encoded label, series uid and center of chunk.
        """
        candidate_info_tup = self.candidate_info_list[index]
        width_irc = (32, 48, 48)

        candidate_a, center_irc = get_ct_raw_candidate(
            candidate_info_tup.series_uid,
            candidate_info_tup.center_xyz,
            width_irc
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)  # Add channel dimension

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