from semantic_parsing.data_utils.data_utils import load_datasets, index_datasets
from semantic_parsing.evaluate import evaluate
from semantic_parsing.parsers.NearestNeighbour import NearestNeighborSemanticParser
from semantic_parsing.parsers.Seq2SeqSemanticParser import Seq2SeqSemanticParser
from semantic_parsing.parsers.Seq2SeqAttentionSP import Seq2SeqAttentionSemanticParser
import semantic_parsing.semantic_parser_config as parser_config
from common.utils.embedding import WordEmbedding
import common.common_config as common_conf
import pickle

import argparse
import numpy as np
import random
import os


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='main.py')

    # General system running and configuration options
    parser.add_argument('--parser', default=parser_config.parser, help='choose the parser')
    parser.add_argument('--train_path', type=str, default=parser_config.data_path + '/geo_train.tsv',
                        help='path to train data')
    parser.add_argument('--dev_path', type=str, default=parser_config.data_path + '/geo_dev.tsv',
                        help='path to dev data')
    parser.add_argument('--test_path', type=str, default=parser_config.data_path + '/geo_test.tsv',
                        help='path to blind test data')
    parser.add_argument('--test_output_path', type=str, default=parser_config.output_path,
                        help='path to write blind test results')
    parser.add_argument('--domain', type=str, default='geo',
                        help='domain (geo for geoquery)')

    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=100, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')

    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data

    train, dev, test = load_datasets(args.train_path, args.dev_path, args.test_path, domain=args.domain)
    train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = \
        index_datasets(train, dev, test, args.decoder_len_limit)

    # Baseline Parser using Nearest Neighborhood matching
    if args.parser == 'NN':
        decoder = NearestNeighborSemanticParser(train_data_indexed)
        evaluate(dev_data_indexed, decoder)

    # Simple Seq2Seq Parser
    elif args.parser == 'Seq2Seq':
        if os.path.isfile(common_conf.encoder_embed_cache):
            print(" Loading cached embedding index ")
            with open(common_conf.encoder_embed_cache, 'rb') as handle:
                ip_embed = pickle.load(handle)

        if os.path.isfile(common_conf.decoder_embed_cache):
            print(" Loading cached embedding index ")
            with open(common_conf.decoder_embed_cache, 'rb') as handle:
                op_embed = pickle.load(handle)
        else:
            print(" creating embed ix and caching ")
            ip_embed = WordEmbedding(pre_trained_embedding_filename=common_conf.glove,
                                     word_indexer=input_indexer)
            op_embed = WordEmbedding(pre_trained_embedding_filename=common_conf.glove,
                                     word_indexer=output_indexer)
            with open(common_conf.encoder_embed_cache, 'wb') as handle:
                pickle.dump(ip_embed, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(common_conf.decoder_embed_cache, 'wb') as handle:
                pickle.dump(op_embed, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

        decoder = Seq2SeqSemanticParser(training_data=train_data_indexed,
                                        dev_data=dev_data_indexed,
                                        input_ix=input_indexer,
                                        output_ix=output_indexer,
                                        ip_embed=ip_embed,
                                        op_embed=op_embed)
        evaluate(dev_data=dev_data_indexed, decoder=decoder)
        # evaluate(dev_data=test_data_indexed, decoder=decoder)
    elif args.parser == 'Seq2SeqAttention':
        decoder = Seq2SeqAttentionSemanticParser(training_data=train_data_indexed,
                                                 dev_data=dev_data_indexed,
                                                 input_ix=input_indexer,
                                                 output_ix=output_indexer)
        evaluate(dev_data=dev_data_indexed, decoder=decoder)
    else:
        raise NotImplementedError
    print("=======FINAL EVALUATION ON BLIND TEST=======")
    evaluate(test_data_indexed, decoder, outfile="geo_test_output.tsv")
