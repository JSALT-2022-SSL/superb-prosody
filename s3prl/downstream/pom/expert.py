import os
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import csv
import json

from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score

from .dataset import POMDataset
from .model import Model

class DownstreamExpert(nn.Module):
    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]
        self.expdir = expdir
        self.logging = os.path.join(expdir, 'log.log')
        self.best = defaultdict(lambda: 0)
        
        self.train_dataset = POMDataset(
            split='train',
            dialogue_dir=self.datarc['dialogue_dir'],
            id_dir=self.datarc['id_dir'],
            label_path=self.datarc['label_path'],
            sample_rate=self.datarc['sample_rate']
        )

        self.dev_dataset = POMDataset(
            split='val',
            dialogue_dir=self.datarc['dialogue_dir'],
            id_dir=self.datarc['id_dir'],
            label_path=self.datarc['label_path'],
            sample_rate=self.datarc['sample_rate']
        )

        self.test_dataset = POMDataset(
            split='test',
            dialogue_dir=self.datarc['dialogue_dir'],
            id_dir=self.datarc['id_dir'],
            label_path=self.datarc['label_path'],
            sample_rate=self.datarc['sample_rate']
        )

        self.connector = nn.Linear(upstream_dim, self.modelrc["projector_dim"])

        self.model = Model(
            input_dim=self.modelrc["projector_dim"],
            output_dim=self.modelrc["output_dim"]
        )

        self.objective = nn.CrossEntropyLoss() 
    
    def get_dataloader(self, mode):
        if mode == "train":
            return self._get_train_dataloader(self.train_dataset)
        elif mode == "val":
            return self._get_eval_dataloader(self.dev_dataset)
        elif mode == "test":
            return self._get_eval_dataloader(self.test_dataset)

    def _get_train_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=True, num_workers=self.datarc['num_workers'],
            drop_last=False, pin_memory=True, collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            drop_last=False, pin_memory=True, collate_fn=dataset.collate_fn
        )

    # Interface
    def forward(self, mode, features, labels, records, **kwargs):
        features = torch.stack(features)
        features = self.connector(features)
        labels = torch.LongTensor(labels).to(features.device)
        # Mean Pooling
        features = torch.mean(features, dim=1)

        # shape: (batch_size, 2)
        predicted = self.model(features)

        loss = self.objective(predicted, labels)

        predicted_class = torch.argmax(predicted, dim=-1)
        
        records['acc'] += (predicted_class == labels).view(-1).cpu().float().tolist()
        records['loss'] += [loss.item()]
        records['predicted_class'] += predicted_class.cpu().float().tolist()
        records['labels'] += labels.cpu().float().tolist()

        del features, labels, predicted, predicted_class

        return loss
        
    def log_records(self, mode, records, logger, global_step,
                    batch_ids, total_batch_num, **kwargs):
        
        macro_f1 = f1_score(records['labels'], records['predicted_class'], average='macro')
        micro_f1 = f1_score(records['labels'], records['predicted_class'], average='micro')

        prefix = f'POM/{mode}-'
        average_acc = torch.FloatTensor(records["acc"]).mean().item()
        average_loss = torch.FloatTensor(records["loss"]).mean().item()

        logger.add_scalar(
            f'{prefix}acc',
            average_acc,
            global_step=global_step
        )

        logger.add_scalar(
            f'{prefix}loss',
            average_loss,
            global_step=global_step
        )

        message = f'mode: {mode}, step: {global_step}, average_acc: {average_acc}, average_loss: {average_loss}, micro_f1: {micro_f1}, macro_f1: {macro_f1}\n'

        save_ckpt = []
        if mode == 'test' and average_acc > self.best[prefix]:
            self.best[prefix] = average_acc
            message = f'best | {message}'
            name = prefix.split('/')[-1].split('-')[0]
            save_ckpt.append(f'best-testing-states-{name}.ckpt')
        
        with open(self.logging, 'a') as f:
            f.write(message)
        
        # print('-'*10, flush=True)
        print(message)
        # print('-'*10, flush=True)

        return save_ckpt
