#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from classifiers.text_classifier import TextClassifier

# FIXME: argparse を使う
DATA_PATH = 'tmp/data.tsv'
RESULT_DIR = 'results'

def main():
    x_train, y_train = read_csv(DATA_PATH)
    x_train, x_text, y_train, y_test = train_test_split(x_train, y_train)
    clf = TextClassifier(path=RESULT_DIR, verbose=True)
    clf.fit(x_train, y_train, time_limit=12*60*60)
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

if __name__ == '__main__':
    main()
