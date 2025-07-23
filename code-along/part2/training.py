import argparse
import datetime
import numpy as np

import torch
import torch.cuda
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from util.util import enumerate_with_estimate
from util.logconf import logging
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

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.total_training_samples_count = 0

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
            is_val_set = False
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
            is_val_set=True
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

    def main(self):
        log.info(f'Starting {type(self).__name__}, {self.cli_args}')

        train_dl = self.init_train_dataloader()
        val_dl = self.init_val_dataloader()

        for epoch in range(1, self.cli_args.epochs + 1):
            train_metrics = self.train(epoch, train_dl)
            self.log_metrics(epoch, 'train', train_metrics)

            val_metrics = self.eval(epoch, val_dl)
            self.log_metrics(epoch, 'eval', val_metrics)

    def train(self, epoch, dataloader):
        self.model.train()

        train_metrics = torch.zeros(
            METRICS_SIZE,
            len(dataloader.dataset),
            device=self.device
        )

        batch_iter = enumerate_with_estimate(
            dataloader,
            f'E{epoch} Training',
            start_index=dataloader.num_workers
        )

        for batch_index, batch_tuple in batch_iter:
            self.optimizer.zero_grad()

            train_loss = self.compute_batch_loss(batch_index, batch_tuple, dataloader.batch_size, train_metrics)
            train_loss.backward()
            self.optimizer.step()

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

            batch_iter = enumerate_with_estimate(
                dataloader,
                f'{epoch} Validation',
                start_index=dataloader.num_workers
            )

            for batch_index, batch_tuple in batch_iter:
                self.compute_batch_loss(batch_index, batch_tuple, dataloader.batch_size, val_metrics)

        return val_metrics.to('cpu')

    def compute_batch_loss(self, batch_index, batch_tuple, batch_size, metrics):
        input_t, label_t, _series_list, _center_list = batch_tuple

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits, probability = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none')       # reduction='none' gives loss per sample
        loss = loss_func(logits, label_g[:,1])

        start_index = batch_index * batch_size
        end_index = start_index + label_t.size(0)

        metrics[METRICS_LABEL_INDEX, start_index:end_index] = label_g[:, 1].detach()
        metrics[METRICS_PRED_INDEX, start_index:end_index] = probability[:, 1].detach()
        metrics[METRICS_LOSS_INDEX, start_index:end_index] = loss.detach()

        return loss.mean()

    @staticmethod
    def log_metrics(epoch_index, mode_str, metrics, classification_threshold=0.5):
        neg_label_mask = metrics[METRICS_LABEL_INDEX] <= classification_threshold
        neg_pred_mask = metrics[METRICS_PRED_INDEX] <= classification_threshold

        pos_label_mask = torch.tensor(~neg_label_mask)
        pos_pred_mask = ~neg_pred_mask

        neg_count = int(neg_label_mask.sum())
        pos_count = int(pos_label_mask.sum())

        neg_correct = int((neg_label_mask & neg_pred_mask).sum())
        pos_correct = int((pos_label_mask & pos_pred_mask).sum())

        metrics_dict = {
            'loss/all': metrics[METRICS_LOSS_INDEX].mean(),
            'loss/neg': metrics[METRICS_LOSS_INDEX, neg_label_mask].mean(),
            'loss/pos': metrics[METRICS_LOSS_INDEX, pos_label_mask].mean(),
            'correct/all': (pos_correct + neg_correct) / np.float32(metrics.shape[1]) * 100,
            'correct/neg': neg_correct / np.float32(neg_count) * 100,
            'correct/pos': pos_correct / np.float32(pos_count) * 100
        }

        log.info(
            f'E{epoch_index} {mode_str:8} {metrics_dict["loss/all"]:.4f} loss, {metrics_dict["correct/all"]:-5.1f}%'
        )

        log.info(
            f'E{epoch_index} {mode_str + "_neg":8}{metrics_dict["loss/neg"]:.4f} loss' +
            f'{metrics_dict["correct/neg"]:-5.1f}% correct ({neg_correct:} of {neg_count:})'
        )

        log.info(
            f'E{epoch_index} {mode_str + "_pos":8}{metrics_dict["loss/pos"]:.4f} loss' +
            f'{metrics_dict["correct/pos"]:-5.1f}% correct ({pos_correct:} of {pos_count:})'
        )


if __name__ == '__main__':
    LunaTrainingApp().main()