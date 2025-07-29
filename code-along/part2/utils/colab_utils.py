import os
import subprocess
import shutil
import concurrent.futures

from .logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

drive_path = f'drive/MyDrive/deep-learning-with-pytorch/data'
local_path = f'data'

def move_file(file_name):
    log.info(f'Moving file {file_name}')
    drive_file = os.path.join(drive_path, file_name)

    shutil.copy(drive_file, local_path)

def delete_directory(dir_name):
    shutil.rmtree(os.path.join(local_path, dir_name))

def move_and_unzip_file(file_name):
    local_zip_file = os.path.join(local_path, file_name)

    if not os.path.exists(local_zip_file):
        move_file(file_name)

    log.info(f'Unzipping file {file_name}')

    subprocess.run([
        'unzip',
        '-q', local_zip_file,
        '-d', local_path
    ])

    log.info(f'Removing file {file_name}')

    os.remove(local_zip_file)

def _execute_subset_load(subset_list, num_workers=None):
    log.info(f'Loading subsets {subset_list}')
    subset_list = [subset for subset in subset_list if not os.path.exists(os.path.join(local_path, subset))]
    num_workers = len(subset_list) if num_workers is None else num_workers
    if len(subset_list) > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(move_and_unzip_file, f'{subset}.zip') for subset in subset_list
            ]
            concurrent.futures.wait(futures)
    log.info(f'Finished loading subset list')


def fetch_data(subset=None):
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    if subset is None:
        # Get all subsets
        missing_subsets = [f'subset{subset_index}' for subset_index in range(10)]
        _execute_subset_load(missing_subsets)
    elif len(subset) > 1:
        # Get all subsets in list in parallel
        subset_list = [f'subset{subset_index}' for subset_index in subset]
        _execute_subset_load(subset_list)
    else:
        move_and_unzip_file(f'subset{subset}.zip')


    move_file('annotations.csv')
    move_file('candidates.csv')

