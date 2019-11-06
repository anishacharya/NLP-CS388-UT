from semantic_parsing.data_utils.definitions import Example, Derivation
import semantic_parsing.semantic_parser_config as parser_conf
from semantic_parsing.data_utils.data_utils import get_xy

from common.utils.indexer import Indexer
from common.utils.embedding import WordEmbedding
from common.Seq2Seq.RNNSeq2Seq import RNNEncoder, RNNDecoder, RNNSeq2Seq
import common.common_config as common_conf

from typing import List
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np


class Seq2SeqSemanticParser(object):
    def __init__(self, training_data: List[Example],
                 dev_data: List[Example],
                 input_ix: Indexer,
                 output_ix: Indexer,
                 ip_embed: WordEmbedding,
                 op_embed: WordEmbedding):
        self.train_data = training_data
        self.dev_data = dev_data
        self.input_ix = input_ix
        self.output_ix = output_ix
        self.encoder_embed = ip_embed
        self.decoder_embed = op_embed
        self.model = self.train()

    def train(self):
        acc = 0.0
        last_epoch_acc = 0.0
        lr = parser_conf.initial_lr
        lr_decay = parser_conf.lr_decay
        weight_decay = parser_conf.weight_decay

        epochs = parser_conf.epochs
        batch_size = parser_conf.batch_size

        encoder = RNNEncoder(conf=parser_conf, word_embed=self.encoder_embed)
        decoder = RNNDecoder(conf=parser_conf, word_embed=self.decoder_embed)
        model = RNNSeq2Seq(encoder=encoder, decoder=decoder)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_function = nn.CrossEntropyLoss(ignore_index=self.encoder_embed.word_ix.
                                            add_and_get_index(common_conf.PAD_TOKEN))

        for epoch in range(0, epochs):
            epoch_loss = 0
            for data_point in self.train_data:
                x, y = get_xy([data_point])
                optimizer.zero_grad()
                y_pred = model(x=x, y=y, teacher_forcing=0.5)
                y_pred = y_pred.view(-1, y_pred.shape[-1])
                y_true = y.view(-1)

                loss = loss_function(y_pred, y_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                epoch_loss += loss.item()
            print(epoch_loss/len(self.train_data))
        return model

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        # Implement the inference here
        raise Exception("implement me!")
