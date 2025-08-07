import os
import argparse
import datetime
import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from utils import logging
from dsets import LunaDataset
from model import LunaModel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

METRICS_LABEL_INDEX = 0
METRICS_PRED_INDEX  = 1
METRICS_LOSS_INDEX  = 2
METRICS_SIZE        = 3

class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys_argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--num_workers',
            help = 'Number of worker processes for background data loading',
            default = 8,
            type = int
        )

        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=32,
            type=int,
        )

        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )

        parser.add_argument(
            '--tb-prefix',
            default='p2ch11',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument(
            '--no-require-on-disk',
            action='store_false',
            help="Require to have the dataset on disk",
            dest='require_on_disk'
        )

        parser.add_argument(
            '--balanced',
            help="Balance the training data to half positive, half negative.",
            action='store_true',
            default=False
        )

        parser.add_argument(
            '--augmented',
            help="Augment the training data.",
            action='store_true',
            default=False,
        )

        parser.add_argument(
            '--augment-flip',
            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
            action='store_true',
            default=False,
        )

        parser.add_argument(
            '--augment-offset',
            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
            action='store_true',
            default=False,
        )

        parser.add_argument(
            '--augment-scale',
            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
            action='store_true',
            default=False,
        )

        parser.add_argument(
            '--augment-rotate',
            help="Augment the training data by randomly rotating the data around the head-foot axis.",
            action='store_true',
            default=False,
        )

        parser.add_argument(
            '--augment-noise',
            help="Augment the training data by randomly adding noise to the data.",
            action='store_true',
            default=False,
        )

        parser.add_argument(
            'comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='LunaTrain',
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.train_writer = None
        self.eval_writer = None
        self.total_training_samples_count = 0

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

    def init_model(self):
        model = LunaModel()
        if self.use_cuda:
            log.info(f'Using CUDA; {torch.cuda.device_count()} devices.')
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def init_optimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def init_train_dataloader(self):
        train_ds = LunaDataset(
            val_stride = 10,
            is_val_set = False,
            ratio_int = int(self.cli_args.balanced),
            augmentation_dict = self.augmentation_dict,
            require_on_disk=self.cli_args.require_on_disk
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size = batch_size,
            num_workers = self.cli_args.num_workers,
            pin_memory = self.use_cuda
        )

        return train_dl

    def init_val_dataloader(self):
        val_ds = LunaDataset(
            val_stride=10,
            is_val_set=True,
            require_on_disk=self.cli_args.require_on_disk
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

    def init_tensorboard_writers(self):
        log.info(f'Initializing tensorboard writers.')
        if self.train_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.train_writer = SummaryWriter(
                log_dir=log_dir + '-train_cls-' + self.cli_args.comment
            )
            self.eval_writer = SummaryWriter(
                log_dir=log_dir + '-train_cls-' + self.cli_args.comment
            )

    def main(self):
        log.info(f'Starting {type(self).__name__}, {self.cli_args}')

        # if get_mode() == 'colab':
        #    fetch_data()

        train_dl = self.init_train_dataloader()
        val_dl = self.init_val_dataloader()

        for epoch in range(1, self.cli_args.epochs + 1):
            log.info(
                f'Epoch {epoch} of {self.cli_args.epochs}, {len(train_dl)}/{len(val_dl)} ' +
                f'batches of size {self.cli_args.batch_size}*{torch.cuda.device_count() if self.use_cuda else 1}'
            )

            train_metrics = self.train(epoch, train_dl)
            self.log_metrics(epoch, 'train', train_metrics)

            val_metrics = self.eval(epoch, val_dl)
            self.log_metrics(epoch, 'eval', val_metrics)

        if hasattr(self, 'train_writer'):
            self.train_writer.close()
            self.eval_writer.close()

    def train(self, epoch, dataloader):
        self.model.train()

        train_metrics = torch.zeros(
            METRICS_SIZE,
            len(dataloader.dataset),
            device=self.device
        )

        train_progress = tqdm(dataloader, desc=f'E{epoch} Training', total=len(dataloader))
        for batch_index, batch_tuple in enumerate(train_progress):
            self.optimizer.zero_grad()

            train_loss = self.compute_batch_loss(batch_index, batch_tuple, dataloader.batch_size, train_metrics)

            train_loss.backward()
            self.optimizer.step()

            # # This is for adding the model graph to TensorBoard.
            # if epoch == 1 and batch_index == 0:
            #     with torch.no_grad():
            #         model = LunaModel()
            #         self.train_writer.add_graph(model, batch_tuple[0], verbose=True)
            #         self.train_writer.close()

        self.total_training_samples_count += len(dataloader.dataset)
        return train_metrics.to('cpu')

    def eval(self, epoch, dataloader):
        with torch.no_grad():
            self.model.eval()

            val_metrics = torch.zeros(
                METRICS_SIZE,
                len(dataloader.dataset),
                device=self.device
            )

            val_progress = tqdm(dataloader, desc=f'E{epoch} Validation', total=len(dataloader))
            for batch_index, batch_tuple in enumerate(val_progress):
                self.compute_batch_loss(batch_index, batch_tuple, dataloader.batch_size, val_metrics)

        return val_metrics.to('cpu')

    def compute_batch_loss(self, batch_index, batch_tuple, batch_size, metrics):
        # log.debug(f'Calculating loss for batch {batch_index} on device {self.device}')

        input_t, label_t, _series_list, _center_list = batch_tuple

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits, probability = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none')       # reduction='none' gives loss per sample
        loss = loss_func(logits, label_g[:, 1])

        start_index = batch_index * batch_size
        end_index = start_index + label_t.size(0)

        metrics[METRICS_LABEL_INDEX, start_index:end_index] = label_g[:, 1].detach()
        metrics[METRICS_PRED_INDEX, start_index:end_index] = probability[:, 1].detach()
        metrics[METRICS_LOSS_INDEX, start_index:end_index] = loss.detach()

        return loss.mean()

    def log_metrics(self, epoch_index, mode_str, metrics, classification_threshold=0.5):
        neg_label_mask: torch.Tensor = metrics[METRICS_LABEL_INDEX] <= classification_threshold
        neg_pred_mask = metrics[METRICS_PRED_INDEX] <= classification_threshold

        pos_label_mask: torch.Tensor = ~neg_label_mask
        pos_pred_mask = ~neg_pred_mask

        neg_count = int(neg_label_mask.sum())
        pos_count = int(pos_label_mask.sum())

        # True negatives: Instances that are predicted as non-nodule and are actually non-nodule
        neg_correct = int((neg_label_mask & neg_pred_mask).sum())
        # True positives: Instances that are predicted as nodules and are actually nodules
        true_pos_count = pos_correct = int((pos_label_mask & pos_pred_mask).sum())

        # Number of instances that are counted as nodules, but are actually non-nodule
        false_pos_count = neg_count - neg_correct
        # Number of instances that are counted as non-nodules, but are actually nodules
        false_neg_count = pos_count - pos_correct

        metrics_dict = {
            'loss/all': metrics[METRICS_LOSS_INDEX].mean(),
            'loss/neg': metrics[METRICS_LOSS_INDEX, neg_label_mask].mean(),
            'loss/pos': metrics[METRICS_LOSS_INDEX, pos_label_mask].mean(),
            'correct/all': (pos_correct + neg_correct) / np.float32(metrics.shape[1]) * 100,
            'correct/neg': neg_correct / np.float32(neg_count) * 100,
            'correct/pos': pos_correct / np.float32(pos_count) * 100
        }

        # Precision: Only classify as true if really sure, minimize the number of false positives
        # How many predicted positives are actually positives
        precision = metrics_dict['pr/precision'] = true_pos_count / np.float32(true_pos_count + false_pos_count)

        # Recall: Maximize the number of interesting events, minimize the number of false negatives
        # How many of the actual positives where classified as positive
        recall = metrics_dict['pr/recall'] = true_pos_count / np.float32(true_pos_count + false_neg_count)

        # F1 Score: ranges between 0 and 1, with 1 being perfect
        metrics_dict['pr/f1_score'] = 2 * (precision / recall) / (precision + recall)

        # Log losses and correct classifications
        log.info(
            f'E{epoch_index} {mode_str:8} {metrics_dict["loss/all"]:.4f} loss, {metrics_dict["correct/all"]:-5.1f}%'
        )

        # Log precision, recall and f1 score
        log.info(
            f'E{epoch_index} {mode_str:8} {metrics_dict["pr/precision"]} precision, ' +
            f'{metrics_dict["pr/recall"]} recall, {metrics_dict["pr/f1_score"]} f1 score'
        )

        # Log number of correctly classified negatives
        log.info(
            f'E{epoch_index} {mode_str + "_neg":8} {metrics_dict["loss/neg"]:.4f} loss ' +
            f'{metrics_dict["correct/neg"]:-5.1f}% correct ({neg_correct:} of {neg_count:})'
        )

        # Log number of correctly classified positives
        log.info(
            f'E{epoch_index} {mode_str + "_pos":8} {metrics_dict["loss/pos"]:.4f} loss ' +
            f'{metrics_dict["correct/pos"]:-5.1f}% correct ({pos_correct:} of {pos_count:})'
        )

        self.init_tensorboard_writers()
        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.total_training_samples_count)

        writer.add_pr_curve(
            'pr',
            metrics[METRICS_LABEL_INDEX],
            metrics[METRICS_PRED_INDEX],
            self.total_training_samples_count,
        )

        bins = [x / 50.0 for x in range(51)]

        neg_hist_mask = neg_label_mask & (metrics[METRICS_PRED_INDEX] > 0.01)
        pos_hist_mask = pos_label_mask & (metrics[METRICS_PRED_INDEX] < 0.99)

        if neg_hist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics[METRICS_PRED_INDEX, neg_hist_mask],
                self.total_training_samples_count,
                bins=bins,
            )
        if pos_hist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics[METRICS_PRED_INDEX, pos_hist_mask],
                self.total_training_samples_count,
                bins=bins,
            )


if __name__ == '__main__':
    LunaTrainingApp().main()