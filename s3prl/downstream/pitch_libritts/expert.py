# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the phone linear downstream wrapper ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
from turtle import color
import torch
import random
import pathlib
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
#-------------#
from ..model import *
from .dataset import PitchDataset
from argparse import Namespace
from pathlib import Path

SAMPLE_RATE = 16000

DEBUG = False
USEBIN = False
NEXT_FRAME = 6
class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, upstream_rate, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.upstream_rate = upstream_rate
        self.downstream = downstream_expert
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir

        root_dir = Path(self.datarc['file_path'])

        self.train_dataset = PitchDataset('train', root_dir, self.datarc['meta_data'], upstream_rate=upstream_rate)
        self.dev_dataset = PitchDataset('dev', root_dir, self.datarc['meta_data'], upstream_rate=upstream_rate)
        self.test_dataset = PitchDataset('test', root_dir, self.datarc['meta_data'], upstream_rate=upstream_rate)
        
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})

        # self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        print("Upstream dimension: ", upstream_dim)
        self.model = model_cls(
            input_dim = upstream_dim,
            output_dim = 33 if USEBIN else 1,
            **model_conf,
        )

        # Fair Experiment
        # self.model = model_cls(
        #     input_dim = upstream_dim,
        #     hiddens = [5],
        #     output_dim = 1,
        #     **model_conf,
        # )

        # Linear
        # self.loss_func = SimpleMSELoss()

        # Normalize
        # mean, std = self.train_dataset.norm_stat
        # self.loss_func = NormalizedMSELoss(mean, std)

        # Log
        self.loss_func = LogMSELoss()

        if USEBIN:
            self.loss_func = nn.CrossEntropyLoss(ignore_index=0)

        self.register_buffer('best_loss', torch.ones(1) * float('inf'))
        if USEBIN:
            self.register_buffer('best_acc', torch.ones(1) * float('-inf'))

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'], 
            shuffle=False, sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )
        # (sampler is None)

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features])
        labels = [feat[:l] for feat, l in zip(labels, features_len)]

        # Sequence mask
        max_len = max(features_len)
        mask = torch.arange(max_len).expand(len(features_len), max_len) < features_len.unsqueeze(1)
        mask = mask.unsqueeze(2)

        features_len = features_len.to(device=device)
        mask = mask.to(device=device)

        features = pad_sequence(features, batch_first=True)
        labels = pad_sequence(labels, batch_first=True).to(device=device)

        if NEXT_FRAME > 0:  # shift labels
            # print(labels.shape)
            labels = torch.roll(labels, -NEXT_FRAME, 1)
            labels[:, -NEXT_FRAME:] = 0

        # Origin
        # features = self.projector(features)
        predicted, _ = self.model(features, features_len)

        if DEBUG:
            print(mask.shape)
            print(features_len)
            print(predicted.shape, labels.shape)

        if not USEBIN:
            # Remove undefined frames
            nan_detect = torch.sum(features, dim=-1, keepdim=True)
            well_defined_mask = torch.logical_and((labels != 0), (nan_detect == nan_detect))
            mask = mask * well_defined_mask

            loss = self.loss_func(predicted, labels, mask)
        else:
            labels = labels.squeeze(-1).long()
            loss = self.loss_func(predicted.transpose(1, 2), labels)
            denom = torch.sum(labels != 0)
            numer = torch.sum((predicted.argmax(dim=2) == labels) * mask.squeeze(-1))
            acc = numer / denom
        if DEBUG:
            print(loss)
            if USEBIN:
                print(acc)
        
        if torch.isfinite(loss):
            if mode == "test":
                self.__vis_prediction(predicted[0][:features_len[0]], labels[0][:features_len[0]], records)
            records['loss'].append(loss.item())
            if USEBIN:
                records['acc'].append(acc.item())
        else:
            loss = torch.zeros(1).to(device=device)
            if DEBUG:
                print("got you!")

        return loss

    def __vis_prediction(self, pred, label, records):
        import numpy as np
        pred = pred.detach().cpu().numpy().squeeze(1)
        pred = np.exp(pred)
        label = label.detach().cpu().numpy().squeeze(1)
        pred[label == 0] = 0
        records["vis"].append((pred, label))

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        keys = ["loss"]
        if USEBIN:
            keys += ["acc"]
        for key in keys:
            average = torch.FloatTensor(records[key]).mean().item()
            logger.add_scalar(
                f'pitch-libritts/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if not USEBIN:
                    if key == 'loss':
                        print(f"{mode} {key}: {average}")
                        f.write(f'{mode} at step {global_step}: {average}\n')
                        if mode == 'dev' and average < self.best_loss:
                            self.best_loss = torch.ones(1) * average
                            f.write(f'New best on {mode} at step {global_step}: {average}\n')
                            save_names.append(f'{mode}-best.ckpt')
                else:
                    if key == 'acc':
                        print(f"{mode} {key}: {average}")
                        f.write(f'{mode} at step {global_step}: {average}\n')
                        if mode == 'dev' and average > self.best_acc:
                            self.best_acc = torch.ones(1) * average
                            f.write(f'New best on {mode} at step {global_step}: {average}\n')
                            save_names.append(f'{mode}-best.ckpt')

        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_loss.txt", "w") as file:
                lines = [f"{x}\n" for x in records["loss"]]
                file.writelines(lines)
            if USEBIN:
                with open(Path(self.expdir) / f"{mode}_acc.txt", "w") as file:
                    lines = [f"{x}\n" for x in records["acc"]]
                    file.writelines(lines)

        # Visualization
        from matplotlib import pyplot as plt
        import numpy as np
        os.makedirs("result/vis-libri-fbank-logmse-yaapt", exist_ok=True)
        if mode in ["test"]:
            for i, (pred, label) in enumerate(records["vis"]):
                plt.plot(np.arange(len(pred)), pred, color='r', label='Prediction')
                plt.plot(np.arange(len(label)), label, color='b', label='Groundtruth')
                plt.legend()
                plt.savefig(f"result/vis-libri-fbank-logmse-yaapt/{i}.png")
                plt.clf()
                if i == 9:
                    break

        return save_names


class SimpleMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pitch_predictions, pitch_targets, mask):
        pitch_predictions = pitch_predictions.masked_select(mask)
        pitch_targets = pitch_targets.masked_select(mask)
        pitch_loss = self.loss(pitch_predictions, pitch_targets)
        return pitch_loss


class NormalizedMSELoss(nn.Module):
    def __init__(self, mean: float, std: float):
        super().__init__()
        self.loss = nn.MSELoss()
        self.mean, self.std, self.eps = mean, std, 1e-6

    def forward(self, pitch_predictions, pitch_targets, mask):
        pitch_predictions = pitch_predictions.masked_select(mask)
        pitch_targets = pitch_targets.masked_select(mask)
        pitch_loss = self.loss(pitch_predictions, (pitch_targets - self.mean) / self.std + self.eps)
        return pitch_loss


class LogMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pitch_predictions, pitch_targets, mask):
        pitch_predictions = pitch_predictions.masked_select(mask)
        pitch_targets = pitch_targets.masked_select(mask)
        pitch_loss = self.loss(pitch_predictions, torch.log(pitch_targets))
        return pitch_loss
