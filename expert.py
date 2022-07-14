from argparse import ArgumentParser, Namespace
from pathlib import Path
from transformers import RobertaTokenizerFast, RobertaModel
from tqdm import trange, tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader

from dataset import MOSEIDataset 
from model import MOSEIdownstream


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def main(args):
    device = args.device
    set_seed(0)
        
    # get upstream model and tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(args.pretrained_model)
    upstream = RobertaModel.from_pretrained(args.pretrained_model)
    upstream = upstream.to(device)

    # get downstream model
    downstream = MOSEIdownstream(upstream.config.hidden_size, args.projection_dim, args.num_classes)
    downstream = downstream.to(device)
    
    # get dataset and dataloader
    trainset = MOSEIDataset(args.data_dir, args.label_path, args.num_classes, args.max_length, tokenizer, mode='train')
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    devset = MOSEIDataset(args.data_dir, args.label_path, args.num_classes, args.max_length, tokenizer, mode='dev')
    dev_loader = DataLoader(devset, batch_size=args.batch_size, shuffle=False)
    testset = MOSEIDataset(args.data_dir, args.label_path, args.num_classes, args.max_length, tokenizer, mode='test')
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    
    # set optimizer and objection function
    optimizer = optim.Adam(downstream.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # set acc and epoch
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_train_acc = 0
    best_dev_acc = 0
    best_test_acc = 0

    # start training
    upstream.eval()
    for epoch in epoch_pbar:
        train_loss = 0
        train_acc = 0
        dev_loss = 0
        dev_acc = 0
        test_loss = 0
        test_acc = 0

        # train
        downstream.train()
        for encode_tokens, labels in tqdm(train_loader):
            encode_tokens = encode_tokens.to(device)
            labels = labels.to(device)
            upstream_outputs = upstream(encode_tokens)   
            cls_token_states = upstream_outputs.last_hidden_state[:,0,:]
            outputs = downstream(cls_token_states)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            train_loss += loss.item() * len(encode_tokens)
            loss.backward()
            optimizer.step()
            _, preds = outputs.max(1)
            train_acc += preds.eq(labels).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            log = f"BEST | Epoch {epoch+1} | train loss: {train_loss} train acc: {train_acc}"
        else:
            log = f"Epoch {epoch+1} | train loss: {train_loss} train acc: {train_acc}"
        print(log)
        with open(args.log_file, 'a') as f:
            f.write(f'{log}\n')

        downstream.eval()
        with torch.no_grad():
            # evaluate
            for encode_tokens, labels in tqdm(dev_loader):
                encode_tokens = encode_tokens.to(device)
                labels = labels.to(device)
                upstream_outputs = upstream(encode_tokens)
                cls_token_states = upstream_outputs.last_hidden_state[:,0,:]
                outputs = downstream(cls_token_states)
                loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                dev_loss += loss.item() * len(encode_tokens)
                _, preds = outputs.max(1)
                dev_acc += preds.eq(labels).sum().item()
            dev_loss /= len(dev_loader.dataset)
            dev_acc /= len(dev_loader.dataset)
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                torch.save(downstream.state_dict(), f'{args.ckpt_dir}/best-dev.pt')
                log = f"BEST | Epoch {epoch+1} | dev loss: {dev_loss} dev acc: {dev_acc}"
            else:
                log = f"Epoch {epoch+1} | dev loss: {dev_loss} dev acc: {dev_acc}"
            print(log)
            with open(args.log_file, 'a') as f:
                f.write(f'{log}\n')

            # test
            for encode_tokens, labels in tqdm(test_loader):
                encode_tokens = encode_tokens.to(device)
                labels = labels.to(device)
                upstream_outputs = upstream(encode_tokens)
                cls_token_states = upstream_outputs.last_hidden_state[:,0,:]
                outputs = downstream(cls_token_states)
                loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                test_loss += loss.item() * len(encode_tokens)
                _, preds = outputs.max(1)
                test_acc += preds.eq(labels).sum().item()
            test_loss /= len(test_loader.dataset)
            test_acc /= len(test_loader.dataset)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(downstream.state_dict(), f'{args.ckpt_dir}/best-test.pt')
                log = f"BEST | Epoch {epoch+1} | test loss: {test_loss} test acc: {test_acc}"
            else:
                log = f"Epoch {epoch+1} | test loss: {test_loss} test acc: {test_acc}"
            print(log)
            with open(args.log_file, 'a') as f:
                f.write(f'{log}\n')
                
            
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./Segmented/Combined/",
    )
    parser.add_argument(
        "--label_path",
        type=Path,
        help="Directory to the dataset.",
        default="./CMU_MOSEI_Labels_new.csv",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        help="Directory to save the model file.",
        default="./log2.log",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        help="Directory to save the model file.",
        default="roberta-base",
    )
    
    # data
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=2)
    
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=50)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
