import jsonlines
import argparse
import numpy as np
from pathlib import Path

def collect_metrics(expname):
    fold_list = []
    for i in range(5):
        fold = f'{expname}-fold-{i}'
        root_dir = Path('result')
        metrics_path = f'result/downstream/{fold}/metrics.jsonl'
    
        with jsonlines.open(metrics_path, 'r') as reader:
            metric_list = [obj for obj in reader]
        fold_list.append(metric_list)

    return fold_list


def cv_score(fold_list):
    best_step, best_precision, best_recall, best_f1 = 0, 0, 0, 0
    for i in range(1, len(fold_list[0]), 2):
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
        
    print('Best Score')
    print(f'step: {best_step}, f1: {best_f1}, precision: {best_precision}, recall: {best_recall}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--expname', help='Save experiment at result/downstream/expname')
    args = parser.parse_args()

    fold_list = collect_metrics(args.expname)
    cv_score(fold_list)


if __name__ == '__main__':
    main()
