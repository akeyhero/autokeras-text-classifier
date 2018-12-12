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
        res = clf.predict([doc])
        print('判定結果: ' + res[0])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=str, default=RESULT_DIR, \
            help='結果出力ディレクトリ (--resume の際はやり直し前後でディレクトリを変更できません)')
    parser.add_argument('--model-prefix', type=str, default=MODEL_PATH_PREFIX, \
            help='モデルファイルプレフィックス')
    return parser.parse_args()

if __name__ == '__main__':
    main()
