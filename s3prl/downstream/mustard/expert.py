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
import jsonlines
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
from sklearn.metrics import classification_report


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
        self.project = self.modelrc['project']
        self.expdir = expdir

        speaker_dependent = self.datarc['speaker_dependent']
        split_no = self.datarc['split_no'] if speaker_dependent else None
        aug_config = downstream_expert.get('augmentation')
        self.train_dataset = SarcasmDataset('train', speaker_dependent, split_no, aug_config)
        self.dev_dataset = SarcasmDataset('dev', speaker_dependent, split_no)
        
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})

        if downstream_expert.get('pca') is not None:
            self.pca = PCA(n_components=downstream_expert['pca']['n_dim'])
            upstream_dim = downstream_expert['pca']['n_dim']
        else:
            self.pca = None

        if self.project:
            self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
            self.model = model_cls(
                input_dim = self.modelrc['projector_dim'],
                output_dim = 1,
                **model_conf,
            )
        else:
            self.model = model_cls(
                input_dim = upstream_dim,
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
        if self.pca is not None:
            features_flat = torch.cat(features, dim=0).detach().cpu().numpy()
            if features_flat.shape[0] > self.pca.n_components:
                self.pca.fit(features_flat)
            features = [torch.from_numpy(self.pca.transform(feature.detach().cpu().numpy())).to(device) for feature in features]

        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        features = pad_sequence(features, batch_first=True)
        if self.project:
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
        result = classification_report(records["truth"], records["predict"], output_dict=True, digits=3)
        metric_obj = {'step': global_step}
        for metric in ['precision', 'recall', 'f1-score']:
            score = result['weighted avg'][metric]
            metric_obj[metric] = score
            print(f'{mode} {metric}: {score}')
            logger.add_scalar(
                f'mustard/{mode}-{metric}',
                score,
                global_step=global_step
            )

            if mode == 'dev' and metric == 'f1-score' and score > self.best_score:
                self.best_score = torch.ones(1) * score
                save_names.append(f'{mode}-best.ckpt')

        if mode == 'dev':
            with jsonlines.open(os.path.join(self.expdir, 'metrics.jsonl'), mode='a') as writer:
                writer.write(metric_obj)

        logger.add_scalar(
            f'mustard/{mode}-loss',
            torch.FloatTensor(records['loss']).mean().item(),
            global_step=global_step
        )                          

        return save_names
