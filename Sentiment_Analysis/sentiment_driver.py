import os
import sys
import argparse
from collections import Counter

sys.path.append(os.path.dirname(__file__) + '../.')

import common.common_config as common_conf
from common.utils.indexer import Indexer
from common.utils.embedding import WordEmbedding

import Sentiment_Analysis.sentiment_config as conf
from Sentiment_Analysis.src.classifiers.ffnn_sentiment_driver import train_sentiment_ffnn
from Sentiment_Analysis.src.data_utils.rotten_tomatoes_reader import (read_and_index_sentiment_examples,
                                                                      write_sentiment_examples)


def _parse_args():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='FF', help='model to run (FF or FANCY)')
    parser.add_argument('--word_vecs_path', type=str, default=common_conf.glove,
                        help='path to word vectors file')
    parser.add_argument('--train_path', type=str, default=conf.data_path + 'train.txt',
                        help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default=conf.data_path + 'dev.txt',
                        help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default=conf.data_path + 'test-blind.txt',
                        help='path to blind test set')
    parser.add_argument('--test_output_path', type=str, default=conf.output_path,
                        help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=conf.run_on_test_flag, action='store_false',
                        help='skip printing output on the test set')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    print(args)

    # initialize an indexer and word counter instance
    # we will update these while reading the data itself.
    word_indexer = Indexer()
    word_indexer.add_and_get_index(common_conf.UNK_TOKEN)

    word_counter = Counter()

    # Load data
    train_data = read_and_index_sentiment_examples(infile=args.train_path,
                                                   word_indexer=word_indexer,
                                                   add_to_indexer=True,
                                                   word_counter=word_counter)
    dev_data = read_and_index_sentiment_examples(infile=args.dev_path,
                                                 word_indexer=word_indexer,
                                                 add_to_indexer=False,
                                                 word_counter=None)
    test_data = read_and_index_sentiment_examples(infile=args.blind_test_path,
                                                  word_indexer=word_indexer,
                                                  add_to_indexer=False,
                                                  word_counter=None)
    print(repr(len(train_data)) + " / " + repr(len(dev_data)) + " / " + repr(len(test_data))
          + " train/dev/test examples")

    # Load Embeddings and create ix2embedding mapping -> Look inside the WordEmbedding class
    word_embedding = WordEmbedding(pre_trained_embedding_filename=common_conf.glove,
                                   word_indexer=word_indexer)

    if args.model == 'FFNN':
        model = train_sentiment_ffnn(train_data=train_data,
                                     dev_data=dev_data,
                                     word_embed=word_embedding)
    else:
        raise NotImplementedError

    if args.run_on_test is True:
        raise NotImplementedError
        # Write the test set output
        # write_sentiment_examples(test_predicted, args.test_output_path, word_vectors.word_indexer)
