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

        self.cli_args = parser.parse_args(sys_argv)
        self.prep_dl = None

    def main(self):
        log.info(f'Starting {type(self).__name__}, {self.cli_args}')

        for subset in range(10):
            fetch_data(subset=subset)

            dataset = LunaDataset()

            self.prep_dl = DataLoader(
                dataset,
                batch_size=self.cli_args.batch_size,
                num_workers=self.cli_args.num_workers,
            )

            subset_progress = tqdm(self.prep_dl, f'Subset {subset}', total=len(self.prep_dl))
            for _ in subset_progress:
                pass

            delete_directory(f'subset{subset}')



if __name__ == '__main__':
    LunaPrepCacheApp().main()