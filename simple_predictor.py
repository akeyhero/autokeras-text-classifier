#!/usr/bin/env python

import argparse

from classifiers.text_classifier import TextClassifier
from build_dataset import get_processor, tokenize # FIXME: utils にする

RESULT_DIR = 'results'
MODEL_PATH_PREFIX = 'tmp/model'

def main():
    args = parse_args()
    processor = get_processor(args.model_prefix)
    clf = TextClassifier(path=args.result_dir, resume=True, verbose=False)
    while True:
        print('判定したい文章: ', end='')
        sentence = input()
        doc = tokenize(sentence, processor)
        print_prediction([doc], clf)

def print_prediction(docs, clf):
    """ 判定結果をプリント; デバッグのため、評価値を表示する """
    x_test = clf.preprocess(docs)
    test_loader = clf.data_transformer.transform_test(x_test)
    scores_array = clf.cnn.predict(test_loader)
    predictions = clf.inverse_transform_y(scores_array)
    for doc, scores, prediction in zip(docs, scores_array, predictions):
        print('判定結果: {} ({}; 「{}」))'.format(prediction, scores, doc))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=str, default=RESULT_DIR, \
            help='結果出力ディレクトリ (--resume の際はやり直し前後でディレクトリを変更できません)')
    parser.add_argument('--model-prefix', type=str, default=MODEL_PATH_PREFIX, \
            help='モデルファイルプレフィックス')
    return parser.parse_args()

if __name__ == '__main__':
    main()
