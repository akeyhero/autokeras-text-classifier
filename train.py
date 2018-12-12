#!/usr/bin/env python

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from classifiers.text_classifier import TextClassifier

DATA_PATH = 'tmp/data.tsv'
RESULT_DIR = 'results'

def main():
    args = parse_args()
    x_train, y_train = read_csv(args.data_path)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train)
    clf = TextClassifier(max_seq_length=args.max_seq_length,
                         path=args.result_dir,
                         resume=args.resume,
                         verbose=True)
    clf.fit(x_train, y_train, time_limit=args.time_limit)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)

def read_csv(file_path):
    """csv file read example method
    It helps you to read the csv file into python array
    Attributes:
        file_path: csv file path
    """
    print("reading data...")
    data_train = pd.read_csv(file_path, sep='\t')
    x_train = np.array(data_train.body)
    y_train = np.array(data_train.label)
    return x_train, y_train

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default=DATA_PATH, \
            help='データセット (id, label, body の TSV) のパス')
    parser.add_argument('--result-dir', type=str, default=RESULT_DIR, \
            help='結果出力ディレクトリ (--resume の際はやり直し前後でディレクトリを変更できません)')
    parser.add_argument('--resume', action='store_true', help='途中からやり直す')
    parser.add_argument('--max-seq-length', type=int, help='最長単語数')
    parser.add_argument('--time-limit', type=int, default=12*60*60, help='学習時間 (s)')
    return parser.parse_args()

if __name__ == '__main__':
    main()
