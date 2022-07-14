import random
import torch
from torch.utils.data.dataset import Dataset
import os
import pandas as pd


class MOSEIDataset(Dataset):
    def __init__(self, data_dir, label_path, num_classes, max_length, tokenizer, mode):
        self.data_dir = data_dir
        self.label_path = label_path
        self.num_classes = num_classes
        self.max_length = max_length
        self.tokenizer = tokenizer
        split_dict = {'train': 0, 'dev': 1, 'test': 2}
        self.split =  split_dict[mode]
        self.encode_tokens_list = []
        self.labels_list = []
        self.get_data()

    def get_data(self):
        data = []
        df = pd.read_csv(self.label_path, encoding="latin-1")
        for row in df.itertuples():
            if row.split ==  self.split:
                filename = row.file + ".txt"
                text_path = os.path.join(self.data_dir, filename)
                with open(text_path, "r") as f:
                    found=False
                    for line in f:
                        elements = line.split('___')
                        if (float(row.start) == float(elements[2])) and (float(row.end) == float(elements[3])):
                            found=True
                            if self.num_classes == 2:
                                data.append((elements[-1], row.label2a))
                            elif self.num_classes == 7:
                                data.append((elements[-1], (row.label7 + 3)))
                            else:
                                print("ERROR: num classes mismatch")
                    if not found:
                        print("ERROR: transcript not found")

        self.encode_tokens_list = self.tokenizer.batch_encode_plus([d[0] for d in data], 
                                                                   max_length=self.max_length,
                                                                   truncation=True,
                                                                   padding="max_length",
                                                                   return_tensors="pt").input_ids
        self.labels_list = [d[1] for d in data]
    
    def __getitem__(self, idx):
        encode_tokens = self.encode_tokens_list[idx]
        labels = self.labels_list[idx]
        return encode_tokens, torch.tensor(labels).long()

    def __len__(self):
        return len(self.labels_list)

