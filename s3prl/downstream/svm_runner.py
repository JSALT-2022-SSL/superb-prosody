import pickle
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import classification_report

from s3prl.downstream.runner import Runner
from s3prl.downstream.mustard.dataset import SarcasmDataset

# Modify run_downstream.py to user this runner
class SVMRunner(Runner):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.init_ckpt = torch.load(self.args.init_ckpt, map_location='cpu') if self.args.init_ckpt else {}
        self.upstream = self._get_upstream()
        self.featurizer = self._get_featurizer()
        
        self.datarc = self.config['downstream_expert']['datarc']
        self.modelrc = self.config['downstream_expert']['modelrc']
        self.expdir = args.expdir
        self.speaker_dependent = self.datarc['speaker_dependent']
        self.split_no = self.datarc['split_no'] if self.speaker_dependent else None 
        self.clf = make_pipeline(
            StandardScaler(),
            svm.SVC(
                C=self.modelrc['svm_c'], 
                gamma="scale", 
                kernel="rbf",
                verbose=True,
            )
        )
    
    def collect_features(self, dataloader, desc):
        feature_list, label_list = [], []
        for batch_id, (wavs, labels, _) in enumerate(tqdm(dataloader, dynamic_ncols=True, desc=desc)):
            wavs = [torch.FloatTensor(wav).to(self.args.device) for wav in wavs]
            with torch.no_grad():
                features = self.upstream.model(wavs)
                features = self.featurizer.model(wavs, features)
                features = [torch.mean(feature, axis=0).detach().cpu().numpy() for feature in features]
            feature_list.append(features)
            label_list += list(labels)

        features = np.concatenate(feature_list, axis=0)
        labels = np.array(label_list)
        return features, labels
    
    def train(self):
        dataset = SarcasmDataset('train', self.speaker_dependent, self.split_no)
        dataloader = DataLoader(
            dataset,
            batch_size=self.datarc['train_batch_size'],
            shuffle=False,
            collate_fn=dataset.collate_fn
        )
        features, labels = self.collect_features(dataloader, 'train')
        self.clf.fit(features, labels)
        with open(os.path.join(self.expdir, 'model.p'), 'wb') as f:
            pickle.dump(self.clf, f)
        self.evaluate()

    def evaluate(self):
        dataset = SarcasmDataset('dev', self.speaker_dependent, self.split_no)
        dataloader = DataLoader(
            dataset,
            batch_size=self.datarc['eval_batch_size'],
            shuffle=False,
            collate_fn=dataset.collate_fn
        )
        features, labels = self.collect_features(dataloader, 'eval')
        preds = self.clf.predict(features)
        result = classification_report(labels, preds, output_dict=True, digits=3)
        metrics = {
            metric: result['weighted avg'][metric]
            for metric in ['precision', 'recall', 'f1-score']
        }
        print(metrics)
        with open(os.path.join(self.expdir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)