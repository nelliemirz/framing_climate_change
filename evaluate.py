import os
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from ast import literal_eval


def detailed_metrics(y_pred, y_true):
    # acc = np.sum(y_pred == y_true) / len(y_pred)
    # recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    # precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    # f1_weighted = f1_score(y_true, y_pred, average='weighted')
    # f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    # correct = sum(preds.flatten() == labels.flatten())

    all_labels = list(set([label for label in y_true]))
    print(all_labels)
    for label in all_labels:
        print(f'f1_w-{label}', f1_score(y_true, y_pred, average='weighted', labels=[label], zero_division=0))
    for label in all_labels:
        print(f'recall-{label}', recall_score(y_true, y_pred, average='weighted', labels=[label], zero_division=0))
    for label in all_labels:
        print(f'precision-{label}',
              precision_score(y_true, y_pred, average='weighted', labels=[label], zero_division=0))


def evaluate(predicted_path,
             gold_path,
             max_count=None):
    assert os.path.exists(gold_path)
    assert os.path.exists(predicted_path)
    if max_count is None:
        with open(gold_path) as gold:
            gold_num_lines = sum(1 for line in gold)
        with open(predicted_path) as pred:
            pred_num_lines = sum(1 for line in pred)
        msg = "Number of lines in files differ: {} vs {}".format(gold_num_lines, pred_num_lines)
        assert gold_num_lines == pred_num_lines, msg


    all_gold = []
    all_pred = []
    with open(gold_path, "r") as gold, open(predicted_path, "r") as pred:
        for i, (y_true, y_pred) in enumerate(zip(gold, pred)):
            y_true = literal_eval(y_true)
            y_pred = literal_eval(y_pred)
            all_gold.append(y_true['labels'])
            all_pred.append(y_pred['labels'])
    detailed_metrics(all_gold, all_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted-path', type=str, required=True)
    parser.add_argument('--gold-path', type=str, required=True)
    parser.add_argument('--max-count', type=int, default=None)
    args = parser.parse_args()
    evaluate(**vars(args))
