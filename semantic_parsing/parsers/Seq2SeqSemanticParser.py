from semantic_parsing.data_utils.definitions import Example, Derivation
import semantic_parsing.semantic_parser_config as parser_conf
from common.utils.indexer import Indexer
from typing import List, Dict
import torch.nn as nn


class Seq2SeqSemanticParser(object):
    def __init__(self, training_data: List[Example],
                 dev_data: List[Example],
                 input_ix: Indexer,
                 output_ix: Indexer,
                 ix2embed: Dict):
        self.train_data = training_data
        self.dev_data = dev_data
        self.input_ix = input_ix
        self.output_ix = output_ix
        self.ix2embed = ix2embed

    def train(self):
        acc = 0.0
        last_epoch_acc = 0.0
        lr = parser_conf.initial_lr
        lr_decay = parser_conf.lr_decay
        weight_decay = parser_conf.weight_decay

        epochs = parser_conf.epochs
        batch_size = parser_conf.batch_size
        loss_function = nn.BCELoss()

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        # Implement the inference here
        raise Exception("implement me!")
