import jsonlines
import json
import argparse
import os
import numpy as np
from pathlib import Path

def collect_metrics(expdir):
    fold_list = []
    for i in range(5):
        fold = f'fold-{i}'
        metrics_path = os.path.join(expdir, f'{fold}/metrics.jsonl')
    
        with jsonlines.open(metrics_path, 'r') as reader:
            metric_list = [obj for obj in reader]
        fold_list.append(metric_list)

    return fold_list


def cv_score(fold_list, expdir):
    best_step, best_precision, best_recall, best_f1 = 0, 0, 0, 0
    for i in range(len(fold_list[0])):
        step = fold_list[0][i]['step']
        precision = np.mean([fold_list[j][i]['precision'] for j in range(5)])
        recall = np.mean([fold_list[j][i]['recall'] for j in range(5)])
        f1 = np.mean([fold_list[j][i]['f1-score'] for j in range(5)])
        # print(f'step: {step}, f1: {f1}, precision: {precision}, recall: {recall}')
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_step = step
        
    result_dict = {'step': best_step, 'f1': best_f1, 'precision': best_precision, 'recall': best_recall}
    print('Best Score')
    print(result_dict)
    with open(os.path.join(expdir, 'metrics.json'), 'w') as f:
        json.dump(result_dict, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--expdir', help='Save experiment at expdir')
    args = parser.parse_args()

    #if args.expdir is None:
    #    args.expdir = f'result/downstream/{args.expname}'

    fold_list = collect_metrics(args.expdir)
    cv_score(fold_list, args.expdir)


if __name__ == '__main__':
    main()