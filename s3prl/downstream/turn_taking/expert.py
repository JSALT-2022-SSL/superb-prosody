import os
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import csv
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .dataset import MaptaskDataset
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
        
        self.train_dataset = MaptaskDataset(
            targets_path = self.datarc["maptask_targets_path"],
            dialogues_path = self.datarc["maptask_dialogues_path"],
            id_list = self.get_table(self.datarc["split_tables"], "train"),
            mode = "train",
            predict_size = (self.modelrc["output_dim"] / 20)
        )

        self.dev_dataset = MaptaskDataset(
            targets_path = self.datarc["maptask_targets_path"],
            dialogues_path = self.datarc["maptask_dialogues_path"],
            id_list = self.get_table(self.datarc["split_tables"], "dev"),
            mode = "dev",
            predict_size = (self.modelrc["output_dim"] / 20)
        )

        self.test_dataset = MaptaskDataset(
            targets_path = self.datarc["maptask_targets_path"],
            dialogues_path = self.datarc["maptask_dialogues_path"],
            id_list = self.get_table(self.datarc["split_tables"], "test"),
            mode = "test",
            predict_size = (self.modelrc["output_dim"] / 20)
        )

        self.connector = nn.Linear(upstream_dim, self.modelrc["projector_dim"])

        self.model = Model(
            input_dim=self.modelrc["projector_dim"] * 2,
            output_dim=self.modelrc["output_dim"],
            dropout=self.modelrc["dropout"],
            hidden_size=self.modelrc["hidden_size"]
        )

        # reference: https://www.cs.utep.edu/nigel/papers/lstm-tt.pdf
        self.objective = nn.BCELoss() 
        
    def get_table(self, table_path, mode):
        with open(f"{table_path}/{mode}_ids.csv", newline='') as csvfile:
            rows = csv.reader(csvfile)
            rows = list(rows)
        id_list = [row[0] for row in rows[1:]]
        return id_list
    
    def get_dataloader(self, mode):
        if mode == "train":
            return self._get_train_dataloader(self.train_dataset)
        elif mode == "dev":
            return self._get_eval_dataloader(self.dev_dataset)
        elif mode == "test":
            return self._get_eval_dataloader(self.test_dataset)

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset,
            batch_size=self.datarc["train_batch_size"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.datarc["eval_batch_size"],
            shuffle=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn
        )

    # Interface
    def forward(self, mode, features, labels, records, **kwargs):
        features = torch.stack(features)
        features = self.connector(features)
        labels = torch.LongTensor(labels).to(features.device)
        labels = labels.float()
        acc = 0
        loss = 0
        
        # mean pooling, shape = (2400, 1, 256)
        features = torch.mean(features, dim=1)

        # split features to g, f, shape = (1200, 256)
        features_g = features[:1200,]
        features_f = features[1200:,]

        # split labels to g, f, shape = (1200, 1)
        labels_g = labels[:,:1200,]
        labels_f = labels[:,1200:,]
        
        # cat features (g, f) predict g, shape = (1, 1200, 512)
        features_cat = torch.unsqueeze(torch.cat((features_g, features_f), dim=-1), 0)        
        predicted = self.model(features_cat)
        predicted = predicted.view(-1)
        labels_g = labels_g.view(-1)
        loss += self.objective(predicted, labels_g)

        predicted_class = (predicted >= 0.5).float()
        acc += (int(predicted_class.eq(labels_g).sum().item()) / predicted_class.shape[0] / 2)

        # cat features (f, g) predict f, shape = (1, 1200, 512)
        features_cat = torch.unsqueeze(torch.cat((features_f, features_g), dim=-1), 0)        
        predicted = self.model(features_cat)
        predicted = predicted.view(-1)
        labels_f = labels_f.view(-1)
        loss += self.objective(predicted, labels_f)

        predicted_class = (predicted >= 0.5).float()
        acc += (int(predicted_class.eq(labels_f).sum().item()) / predicted_class.shape[0] / 2)

        records['acc'] += [acc]
        records['loss'] += [loss]
        
        return loss
        
    def log_records(self, mode, records, logger, global_step,
                    batch_ids, total_batch_num, **kwargs):

        prefix = f'turn_taking/{mode}-'
        average_acc = torch.FloatTensor(records['acc']).mean().item()
        average_loss = torch.FloatTensor(records['loss']).mean().item()
        
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
        message = f'{prefix}|step:{global_step}|acc:{average_acc}|loss:{average_loss}\n'
        save_ckpt = []
        if average_acc > self.best[prefix]:
            self.best[prefix] = average_acc
            message = f'best|{message}'
            name = prefix.split('/')[-1].split('-')[0]
            save_ckpt.append(f'best-states-{name}.ckpt')
        with open(self.logging, 'a') as f:
            f.write(message)
        print(message)
        
        return save_ckpt
