import os
import sys
import shutil
import hashlib
import datetime

import numpy as np

import torch.cuda
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from training import TrainingApp
from unet_model import UNetWrapper, SegmentationAugmentation
from segmentation_ds import get_segmentation_ct, TrainingLuna2dSegmentationDataset, Luna2dSegmentationDataset

from utils import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

METRICS_LOSS_INDEX = 1
METRICS_TP_INDEX = 7
METRICS_FN_INDEX = 8
METRICS_FP_INDEX = 9

METRICS_SIZE = 10

class SegmentationTrainingApp(TrainingApp):
    def __init__(self, sys_argv=None):
        super().__init__(sys_argv)

        self.segmentation_model, self.augmentation_model = self.init_model()
        self.optimizer = self.init_optimizer()
        self.validation_cadence = 5

    def init_model(self):
        segmentation_model = UNetWrapper(
            in_channels=7, n_classes=1, depth=3, wf=4,
            padding=True, batch_norm=True, up_mode='upconv'
        )

        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info(f'Using CUDA; {torch.cuda.device_count()} devices.')

            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            segmentation_model = segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)

        return segmentation_model, augmentation_model

    def init_optimizer(self):
        return Adam(self.segmentation_model.parameters())

    def init_train_dl(self):
        train_ds = TrainingLuna2dSegmentationDataset(
            data_split='train',
            ratios=(0.9, 0.1, 0),
            context_slice_count=3,
            require_on_disk = self.cli_args.require_on_disk
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )
        return train_dl

    def init_val_dl(self):
        val_ds = Luna2dSegmentationDataset(
            data_split='eval',
            ratios=(0.9, 0.1, 0),
            context_slice_count=3
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )
        return val_dl

    @staticmethod
    def dice_loss(predictions, labels, epsilon=1):
        dice_labels = labels.sum(dim=[1, 2, 3])
        dice_predictions = predictions.sum(dim=[1, 2, 3])
        dice_correct = (predictions * labels).sum(dim=[1, 2, 3])

        dice_ratio = (2 * dice_correct + epsilon) \
            / (dice_predictions + dice_labels + epsilon)

        return 1 - dice_ratio

    def compute_batch_loss(self, batch_index, batch_tup, batch_size, metrics, classification_threshold=0.5):
        inputs, labels, series_list, slice_index_list = batch_tup

        inputs = inputs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        if self.segmentation_model.training and self.augmentation_dict:
            inputs, labels = self.augmentation_model(inputs, labels)

        predictions = self.segmentation_model(inputs)

        dice_loss = self.dice_loss(predictions, labels)             # Loss for the whole sample
        fine_loss = self.dice_loss(predictions * labels, labels)    # Loss only for the pixels, where labels is true

        start_index = batch_index * batch_size
        end_index = start_index + inputs.size(0)

        with torch.no_grad():
            predictions_bool = (predictions[:, 0:1] > classification_threshold).to(torch.float32)

            tp = (predictions_bool * labels).sum(dim=[1, 2, 3])
            fn = ((1 - predictions_bool) * labels).sum(dim=[1, 2, 3])
            fp = (predictions_bool * (~labels)).sum(dim=[1, 2, 3])

            metrics[METRICS_LOSS_INDEX, start_index:end_index] = dice_loss
            metrics[METRICS_TP_INDEX, start_index:end_index] = tp
            metrics[METRICS_FN_INDEX, start_index:end_index] = fn
            metrics[METRICS_FP_INDEX, start_index:end_index] = fp

        # The loss function favors the loss of pixels where the label is true
        # favoring positive pixels over negative ones
        # Our goal is to have high recall, i.e. as little false negatives as possible, thus a higher weighting of
        # positive pixels helps with minimizing
        # This will lead to more false positives, however.
        # Our goal is cancer detection, wrongly classifying a negative pixels as nodule is much less of a problem
        # than missing an actual nodule, classifying a positive pixel as negative.
        return dice_loss.mean() + fine_loss.mean() * 8

    def main(self):
        log.info(f'Starting {type(self).__name__, self.cli_args}')

        train_dl = self.init_train_dl()
        val_dl = self.init_val_dl()

        best_score = 0.0
        for epoch_index in range(1, self.cli_args.epochs + 1):
            log.info(
                f'Epoch {epoch_index} of {self.cli_args.epochs}, ' +
                f'{len(train_dl)}/{len(val_dl)} batches of size {self.cli_args.batch_size} '
            )

            train_metrics = self.train(epoch_index, self.segmentation_model, self.optimizer, train_dl, METRICS_SIZE)
            self.log_metrics(epoch_index, 'train', train_metrics)

            if epoch_index == 1 or epoch_index % self.validation_cadence == 0:
                val_metrics = self.eval(epoch_index, self.segmentation_model, val_dl, METRICS_SIZE)
                score = self.log_metrics(epoch_index, 'eval', val_metrics)
                best_score = max(score, best_score)

                self.save_model('seg', epoch_index, score == best_score)

                self.log_images(epoch_index, 'train', train_dl)
                self.log_images(epoch_index, 'eval', val_dl)

        self.writer.close()


    def log_images(self, epoch_index, mode_str, dl):
        # Log 6 evenly spaced slices end-to-end to show the ground truth and our model's output
        self.segmentation_model.eval()

        images = sorted(dl.dataset.series_list)[:12]
        for series_index, series_uid in enumerate(images):
            ct = get_segmentation_ct(series_uid)

            for slice_index in range(6):
                # Select 6 equidistant slices (0 * 24, 1 * 24...) of a ct scan
                ct_index = slice_index * (ct.hu_a.shape[0] - 1) // 5

                sample_tup = dl.dataset.getitem_full_slice(series_uid, ct_index)
                ct_t, labels, series_uid, ct_index = sample_tup

                inputs = ct_t.to(self.device).unsqueeze(0)
                labels = labels.to(self.device).unsqueeze(0)

                predictions = self.segmentation_model(inputs)[0]
                predictions = predictions.to('cpu').numpy()[0] > 0.5
                labels = labels.to('cpu').numpy()[0][0] > 0.5

                ct_t[:-1, :, :] /= 2000
                ct_t[:-1, :, :] += 0.5

                ct_slice = ct_t[dl.dataset.context_slice_count].numpy()

                image_a = np.zeros((512, 512, 3), dtype=np.float32)     # 3 channels for RGB
                image_a[:, :, :] = ct_slice.reshape((512, 512, 1))          # Base image is a grey-scale of ct intensity

                # Mark false classifications as red
                image_a[:, :, 0] += predictions & (1 - labels)

                # Mark false negatives as orange
                image_a[:, :, 0] += (1 - predictions) & labels
                image_a[:, :, 1] += ((1 - predictions) & labels) * 0.5

                # Mark true positives as green
                image_a[:, :, 1] += predictions * labels

                # Normalize the data
                image_a *= 0.5
                image_a.clip(0, 1, image_a)

                self.writer.add_image(
                    f'{mode_str}/{series_index}_prediction_{slice_index}',
                    image_a,
                    self.total_training_samples_count,
                    dataformats='HWC'       # Channels is third axis
                )

                if epoch_index == 1:
                    image_a = np.zeros((512, 512, 3), dtype=np.float32)
                    image_a[:, :, :] = ct_slice.reshape((512, 512, 1))
                    image_a[:, :, 1] += labels

                    image_a *= 0.5
                    image_a[image_a < 0] = 0
                    image_a[image_a > 1] = 1
                    self.writer.add_image(
                        f'{mode_str}/{series_index}_label_{slice_index}',
                        image_a,
                        self.total_training_samples_count,
                        dataformats='HWC'
                    )

                self.writer.flush()

    def log_metrics(self, epoch_index, mode_str, metrics):
        log.info(f'E{epoch_index} {type(self).__name__}')

        metrics = metrics.detach().numpy()
        sum_a = metrics.sum(axis=1)
        assert np.isfinite(metrics).all()

        all_label_count = sum_a[METRICS_TP_INDEX] + sum_a[METRICS_FN_INDEX]

        metrics_dict = {
            f'{mode_str}/loss/all': metrics[METRICS_LOSS_INDEX].mean(),
            f'{mode_str}/percent_all/tp': sum_a[METRICS_TP_INDEX] / (all_label_count or 1) * 100,
            f'{mode_str}/percent_all/fn': sum_a[METRICS_FN_INDEX] / (all_label_count or 1) * 100,
            f'{mode_str}/percent_all/fp': sum_a[METRICS_FP_INDEX] / (all_label_count or 1) * 100
        }

        precision = metrics_dict[f'{mode_str}/pr/precision'] = \
            sum_a[METRICS_TP_INDEX] / ((sum_a[METRICS_TP_INDEX] + sum_a[METRICS_FP_INDEX]) or 1)
        recall = metrics_dict[f'{mode_str}/pr/recall'] = \
            sum_a[METRICS_TP_INDEX] / ((sum_a[METRICS_TP_INDEX] + sum_a[METRICS_FN_INDEX]) or 1)

        metrics_dict[f'{mode_str}/pr/f1_score'] = 2 * (precision * recall) / ((precision + recall) or 1)

        log.info(
            f"E{epoch_index} {mode_str:8} "
            f"{metrics_dict[f'{mode_str}/loss/all']:.4f} loss, "
            f"{metrics_dict[f'{mode_str}/pr/precision']:.4f} precision, "
            f"{metrics_dict[f'{mode_str}/pr/recall']:.4f} recall, "
            f"{metrics_dict[f'{mode_str}/pr/f1_score']:.4f} f1 score"
        )
        log.info(
            f"E{epoch_index} {mode_str + '_all':8} "
            f"{metrics_dict[f'{mode_str}/loss/all']:.4f} loss, "
            f"{metrics_dict[f'{mode_str}/percent_all/tp']:-5.1f}% tp, "
            f"{metrics_dict[f'{mode_str}/percent_all/fn']:-5.1f}% fn, "
            f"{metrics_dict[f'{mode_str}/percent_all/fp']:-9.1f}% fp"
        )

        self.init_tensorboard_writer()

        prefix_str = 'seq_'
        for key, value in metrics_dict.items():
            self.writer.add_scalar(prefix_str + key, value, self.total_training_samples_count)
        self.writer.flush()

        score = metrics_dict[f'{mode_str}/pr/recall']
        return score

    def save_model(self, type_str, epoch_index, is_best=False):
        file_path = os.path.join(
            'data', 'models', self.cli_args.tb_prefix,
            f'{type_str}_{self.time_str}_{self.cli_args.comment}_{self.total_training_samples_count}'
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.segmentation_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_index,
            'total_training_samples_count': self.total_training_samples_count
        }
        torch.save(state, file_path)
        log.info(f'Saved model parameters to {file_path}')

        if is_best:
            best_path = os.path.join(
                'data', 'models', self.cli_args.tb_prefix,
                f'{type_str}_{self.time_str}_{self.cli_args.comment}.best.state'
            )
            shutil.copyfile(file_path, best_path)

            log.info(f'Saved model params to {best_path}')

        with open(file_path, 'rb') as f:
            log.info(f'SHA1: {hashlib.sha1(f.read()).hexdigest()}')


if __name__ == '__main__':
    SegmentationTrainingApp().main()