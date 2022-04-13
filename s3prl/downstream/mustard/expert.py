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
from .dataset import SarcasmDataset
from argparse import Namespace
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir

        self.train_dataset = SarcasmDataset('train', downstream_expert['augmentation'])
        self.dev_dataset = SarcasmDataset('dev')
        self.test_dataset = SarcasmDataset('test')
        
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = 1,
            **model_conf,
        )
        self.objective = nn.BCEWithLogitsLoss()
        self.register_buffer('best_score', torch.zeros(1))

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'], 
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

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
    def forward(self, mode, features, labels, file_ids, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)

        predicted, _ = self.model(features, features_len)
        labels = torch.FloatTensor(labels).to(features.device)
        loss = self.objective(predicted.squeeze(dim=1), labels)

        predicted_classid = torch.round(torch.sigmoid(predicted)).squeeze(dim=1)#predicted.max(dim=-1).indices

        records['loss'].append(loss.item())
        records['filename'] += file_ids
        records['predict'] += predicted_classid.cpu().tolist()
        records['truth'] += labels.cpu().tolist()

        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []

        for key in ['precision', 'recall', 'f1', 'loss']:
            if key != 'loss':
                
                score = eval(f'{key}_score(records["truth"], records["predict"])')
                print(f"{mode} {key}: {score}")

                with open(Path(self.expdir) / "log.log", 'a') as f:
                    f.write(f'{mode} {key} at step {global_step}: {score}\n')
                    if mode == 'dev' and key == 'f1' and score > self.best_score:
                        self.best_score = torch.ones(1) * score
                        f.write(f'New best on {mode} at step {global_step}: {score}\n')
                        save_names.append(f'{mode}-best.ckpt')

            else:
                score = torch.FloatTensor(records[key]).mean().item()

            logger.add_scalar(
                f'mustard/{mode}-{key}',
                score,
                global_step=global_step
            )

        # if mode in ["dev", "test"]:
        #     with open(Path(self.expdir) / f"{mode}_predict.txt", "w") as file:
        #         lines = [f"{f} {p}\n" for f, p in zip(records["filename"], records["predict"])]
        #         file.writelines(lines)

        #     with open(Path(self.expdir) / f"{mode}_truth.txt", "w") as file:
        #         lines = [f"{f} {l}\n" for f, l in zip(records["filename"], records["truth"])]
        #         file.writelines(lines)

        return save_names
