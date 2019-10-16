import os
import sys
import argparse
from collections import Counter
import torch

sys.path.append(os.path.dirname(__file__) + '../.')

import common.common_config as common_conf
from common.utils.indexer import Indexer
from common.utils.embedding import WordEmbedding
from common.models.FFNN import FFNN
from common.models.RNN import RNN

import Sentiment_Classification.sentiment_config as sentiment_conf
from Sentiment_Classification.src.classifiers.ffnn_sentiment_driver import train_sentiment_ffnn
from Sentiment_Classification.src.classifiers.rnn_sentiment_driver import train_sentiment_rnn
from Sentiment_Classification.src.data_utils.rotten_tomatoes_reader import (read_and_index_sentiment_examples,
                                                                            write_sentiment_examples)
from Sentiment_Classification.src.evaluation.evaluate import evaluate_sentiment, evaluate_sentiment_simple
from Sentiment_Classification.src.data_utils.definitions import SentimentExample


def _parse_args():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default=sentiment_conf.model,
                        help='model to run (FF or FANCY)')
    parser.add_argument('--word_vecs_path', type=str, default=common_conf.glove,
                        help='path to word vectors file')
    parser.add_argument('--train_path', type=str, default=sentiment_conf.data_path + 'train.txt',
                        help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default=sentiment_conf.data_path + 'dev.txt',
                        help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default=sentiment_conf.data_path + 'test-blind.txt',
                        help='path to blind test set')
    parser.add_argument('--test_output_path', type=str, default=sentiment_conf.output_path,
                        help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=sentiment_conf.run_on_test_flag,
                        action='store_false', help='skip printing output on the test set')
    parser.add_argument('--no_run_on_manual', dest='run_on_manual', default=sentiment_conf.run_on_manual_flag,
                        action='store_false', help='skip printing output on the manual set')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    print(args)

    # initialize an indexer and word counter instance
    # we will update these while reading the data itself.
    word_indexer = Indexer()
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

    # train_data = train_data[0:10]
    # Load Embeddings and create ix2embedding mapping -> Look inside the WordEmbedding class
    word_embedding = WordEmbedding(pre_trained_embedding_filename=common_conf.glove,
                                   word_indexer=word_indexer)

    if args.model == 'FFNN':
        train_sentiment_ffnn(train_data=train_data,
                             dev_data=dev_data,
                             word_embed=word_embedding)
        model = FFNN(sentiment_conf)
        model.load_state_dict(torch.load(sentiment_conf.model_path))
    elif args.model == 'RNN':
        train_sentiment_rnn(train_data=train_data,
                            dev_data=dev_data,
                            test_data=test_data,
                            word_embed=word_embedding)
        model = RNN(conf=sentiment_conf, word_embed=word_embedding)
        model.load_state_dict(torch.load(sentiment_conf.model_path))
    else:
        raise NotImplementedError
    # _, metrics = evaluate_sentiment(model=model, data=dev_data,
    #                                 word_embedding=word_embedding, model_type=args.model)
    _, accuracy = evaluate_sentiment_simple(model=model, data=dev_data,
                                            word_embedding=word_embedding, model_type=args.model)
    print("Final Dev Accuracy = ", accuracy)

    if args.run_on_manual is True:
        y_pred, _ = evaluate_sentiment_simple(model=model, model_type='RNN',
                                              word_embedding=word_embedding, data=test_data)
        test_predicted = []
        for pred, data_point in zip(y_pred, test_data):
            test_predicted.append(SentimentExample(label=int(pred), indexed_words=data_point.indexed_words))
        # Write the test set output
        print('writing Manual Output')
        write_sentiment_examples(test_predicted, sentiment_conf.output_path, word_embedding.word_ix)
        print('Done Writing Manual Output')
    # if args.run_on_test is True:
    #     # y_pred, _ = evaluate_sentiment(model=model, model_type=args.model,
    #     #                                word_embedding=word_embedding, data=test_data)
    #     y_pred, _ = evaluate_sentiment(model=model, model_type=args.model,
    #                                    word_embedding=word_embedding, data=test_data)
    #     test_predicted = []
    #     for pred, data_point in zip(y_pred, test_data):
    #         test_predicted.append(SentimentExample(label=int(pred), indexed_words=data_point.indexed_words))
    #     # Write the test set output
    #     write_sentiment_examples(test_predicted, args.test_output_path, word_embedding.word_ix)
