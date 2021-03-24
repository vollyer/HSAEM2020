#!/usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_samples(sample_file):
    sample = []
    with open(sample_file, 'r') as f:
        for line in f:
            line_data = line.strip().split(' ')
            sample.append((line_data[0], line_data[1]))
    return sample

def feature(file, oper, pairs):

    # d_vector = pd.read_table(file, header=None, encoding='gb2312', delimiter='\t', index_col=0)
    d_vector = pd.read_table(file, header=None, encoding='gb2312', delim_whitespace=True, index_col=0,
                             skiprows=[0])

    pair_vector = []
    for pair in pairs:
        if oper == 'average':
            if pair[0] == pair[1]:
                continue
            pair_vector.append((d_vector.loc[pair[0]] + d_vector.loc[pair[1]]) / 2.)
        if oper == 'Hadamard':
            if pair[0] == pair[1]:
                continue
            pair_vector.append(d_vector.loc[pair[0]] * d_vector.loc[pair[1]])
        if oper == 'L1':
            if pair[0] == pair[1]:
                continue
            pair_vector.append(abs(d_vector.loc[pair[0]] - d_vector.loc[pair[1]]))
        if oper == 'L2':
            if pair[0] == pair[1]:
                continue

            pair_vector.append(np.square(d_vector.loc[pair[0]] - d_vector.loc[pair[1]]))
    return pair_vector


def load_feature(args):
    train_pos = load_samples(args.input)
    train_neg = load_samples(args.train_neg)
    train_labels = [1] * len(train_pos) + [0] * len(train_neg)
    train_pos.extend(train_neg)
    train_features = feature(args.output, args.oper_name, train_pos)

    test_pos = load_samples(args.test_pos)
    test_neg = load_samples(args.test_neg)
    test_labels = [1] * len(test_pos) + [0] * len(test_neg)
    test_pos.extend(test_neg)
    test_features = feature(args.output, args.oper_name, test_pos)

    return train_features, train_labels, test_features, test_labels

def draw_ROC_curve(test_label, test_prob):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(test_label, test_prob)
    print(auc(false_positive_rate, true_positive_rate))
    plt.plot(false_positive_rate, true_positive_rate, label='(ROC=%0.4f)' % auc(false_positive_rate, true_positive_rate))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title('ROC', fontsize=20)
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.savefig('roc.png')
    plt.show()
