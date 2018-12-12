#!/usr/bin/env python

import argparse

from itertools import cycle, chain
import sentencepiece as spm

ALL_DOCS_PATH       = 'tmp/all.txt'
POS_DOCS_PATH       = 'tmp/positive.txt'
NEG_DOCS_PATH       = 'tmp/negative.txt'
DATA_PATH           = 'tmp/data.tsv'
MODEL_PATH_PREFIX   = 'tmp/model'
VOCAB_SIZE          = 10000
MAX_SENTENCE_LENGTH = 8192

def main():
    args = parse_args()
    build_model(args.all_docs_path, args.model_out_prefix, args.vocab_size, args.max_sentence_length)
    processor = get_processor(args.model_out_prefix)
    with open(args.pos_docs_path) as pos_f, open(args.neg_docs_path) as neg_f:
        print('Tokenizing and writing sentences...')
        positive_sentences = (tokenize(line, processor) for line in pos_f)
        negative_sentences = (tokenize(line, processor) for line in neg_f)
        labeled_sentences = chain(
            zip(cycle(['p']), positive_sentences),
            zip(cycle(['n']), negative_sentences)
        )
        write_with_id(args.data_out, labeled_sentences, header=['label', 'body'])
    print('Done.')

def build_model(_input, model_prefix, vocab_size, max_sentence_length):
    spm.SentencePieceTrainer.Train(' '.join([
        '--input='               + _input,
        '--model_prefix='        + model_prefix,
        '--vocab_size='          + str(vocab_size),
        '--max_sentence_length=' + str(max_sentence_length)
    ]))

def get_processor(model_path_prefix):
    processor = spm.SentencePieceProcessor()
    processor.Load(model_path_prefix + '.model')
    return processor

def write_with_id(path, values_list, header=None, separator="\t"):
    with open(path, mode='w') as f:
        if header is not None:
            f.write(separator.join(['id', *header]))
            f.write("\n")
        for i, values in enumerate(values_list):
            f.write(separator.join(map(str, [i, *values])))
            f.write("\n")

def tokenize(sentence, processor):
    return ' '.join(processor.EncodeAsPieces(sentence))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-docs-path', type=str, default=ALL_DOCS_PATH, \
            help='全文書を列挙したもの')
    parser.add_argument('--pos-docs-path', type=str, default=POS_DOCS_PATH, \
            help='ポジティブな文書のみを列挙したもの')
    parser.add_argument('--neg-docs-path', type=str, default=NEG_DOCS_PATH, \
            help='ネガティブな文書のみを列挙したもの')
    parser.add_argument('--data-out', type=str, default=DATA_PATH, \
            help='データ出力先')
    parser.add_argument('--model-out-prefix', type=str, default=MODEL_PATH_PREFIX, \
            help='モデル出力先プレフィックス')
    parser.add_argument('--vocab-size', type=str, default=VOCAB_SIZE, \
            help='語彙のサイズ')
    parser.add_argument('--max-sentence-length', type=str, default=MAX_SENTENCE_LENGTH, \
            help='最大文書長 (bytes)')
    return parser.parse_args()

if __name__ == '__main__':
    main()
