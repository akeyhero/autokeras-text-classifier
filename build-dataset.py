#!/usr/bin/env python

import sentencepiece as spm
from itertools import cycle, chain

# FIXME: argparse を使う
ALL_DOCS_PATH       = 'tmp/all.txt'
POS_DOCS_PATH       = 'tmp/positive.txt'
NEG_DOCS_PATH       = 'tmp/negative.txt'
DATA_PATH           = 'tmp/data.tsv'
MODEL_PATH_PREFIX   = 'tmp/model'
VOCAB_SIZE          = 10000
MAX_SENTENCE_LENGTH = 8192

def main():
    build_model()
    processor = get_processor()
    with open(POS_DOCS_PATH) as pos_f, open(NEG_DOCS_PATH) as neg_f:
        print('Tokenizing and writing sentences...')
        positive_sentences = (tokenize(line, processor) for line in pos_f)
        negative_sentences = (tokenize(line, processor) for line in neg_f)
        labeled_sentences = chain(
            zip(cycle(['p']), positive_sentences),
            zip(cycle(['n']), negative_sentences)
        )
        write_with_id(DATA_PATH, labeled_sentences, header=['label', 'body'])
    print('Done.')

def build_model():
    spm.SentencePieceTrainer.Train(' '.join([
        '--input='               + ALL_DOCS_PATH,
        '--model_prefix='        + MODEL_PATH_PREFIX,
        '--vocab_size='          + str(VOCAB_SIZE),
        '--max_sentence_length=' + str(MAX_SENTENCE_LENGTH)
    ]))

def get_processor():
    processor = spm.SentencePieceProcessor()
    processor.Load(MODEL_PATH_PREFIX + '.model')
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

if __name__ == '__main__':
    main()
