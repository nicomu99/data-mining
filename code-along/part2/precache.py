import math
import argparse
import sys

from torch.utils.data import DataLoader

from tqdm import tqdm
from utils import fetch_data, delete_directory
from dsets import LunaDataset
from utils import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class LunaPrepCacheApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=32,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--subset-count',
            help='Number of subsets to process at the same time',
            default=3,
            type=int,
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.prep_dl = None
        self.num_subsets = 10
        self.num_subset_batches = math.ceil(self.num_subsets / self.cli_args.subset_count) + 1
        self.ignore_set = {}

    def main(self):
        log.info(f'Starting {type(self).__name__}, {self.cli_args}')

        for subset_batch in range(self.num_subset_batches):
            min_subset_index = subset_batch * self.cli_args.subset_count
            max_subset_index = min(self.num_subsets, (subset_batch + 1) * self.cli_args.subset_count)
            subsets_to_fetch = list(range(min_subset_index, max_subset_index))
            subsets_to_fetch = [subset for subset in subsets_to_fetch if subset not in self.ignore_set]

            fetch_data(subsets_to_fetch)

            dataset = LunaDataset()

            self.prep_dl = DataLoader(
                dataset,
                batch_size=self.cli_args.batch_size,
                num_workers=self.cli_args.num_workers,
            )

            subset_progress = tqdm(self.prep_dl, f'Subsets {subsets_to_fetch}', total=len(self.prep_dl))
            for _ in subset_progress:
                pass

            for subset in subsets_to_fetch:
                delete_directory(f'subset{subset}')

    def add_to_ignore_set(self, subsets_to_add):
        self.ignore_set.update(subsets_to_add)

    def remove_from_ignore_set(self, subsets_to_remove):
        self.ignore_set -= set(subsets_to_remove)

if __name__ == '__main__':
    LunaPrepCacheApp().main()