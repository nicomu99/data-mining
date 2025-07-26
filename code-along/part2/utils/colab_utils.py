import os
import subprocess
import shutil
import concurrent.futures

from util import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

drive_path = f'drive/MyDrive/deep-learning-with-pytorch/data'
local_path = f'data'

def move_file(file_name):
    log.info(f'Moving file {file_name}')
    drive_file = os.path.join(drive_path, file_name)

    shutil.copy(drive_file, local_path)

def move_and_unzip_file(file_name):
    local_zip_file = os.path.join(local_path, file_name)

    move_file(file_name)

    log.info(f'Unzipping file {file_name}')

    subprocess.run([
        'unzip',
        '-q', local_zip_file,
        '-d', local_path
    ])

    log.info(f'Removing file {file_name}')

    shutil.rmtree(local_zip_file)

def fetch_data():
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    missing_subsets = []
    for subset_index in range(10):
        subset = f'subset{subset_index}'
        if not os.path.exists(subset):
            missing_subsets.append(subset)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(move_and_unzip_file, f'{subset}.zip') for subset in missing_subsets
        ]
        concurrent.futures.wait(futures)

    move_file('annotations.csv')
    move_file('candidates.csv')

