from semantic_parsing.data_utils.definitions import Example, Derivation
import semantic_parsing.semantic_parser_config as parser_conf
from semantic_parsing.data_utils.data_utils import get_xy
from semantic_parsing.evaluate import evaluate

from common.utils.indexer import Indexer
from common.utils.embedding import WordEmbedding
from common.Seq2Seq.RNNSeq2SeqAttention import Encoder, Decoder, Seq2Seq
import common.common_config as common_conf
from common.utils.utils import get_batch

from typing import List
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import random


class Seq2SeqAttentionSemanticParser(object):
    def __init__(self, training_data: List[Example],
                 dev_data: List[Example],
                 input_ix: Indexer,
                 output_ix: Indexer):
        self.train_data = training_data
        self.dev_data = dev_data
        self.input_ix = input_ix
        self.output_ix = output_ix
        self.model = self.train()

    def train(self):
        acc = 0.0
        last_epoch_acc = 0.0
        lr = parser_conf.initial_lr
        lr_decay = parser_conf.lr_decay
        weight_decay = parser_conf.weight_decay

        epochs = parser_conf.epochs
        batch_size = parser_conf.batch_size

        encoder = Encoder(conf=parser_conf)
        decoder = Decoder(conf=parser_conf)
        model = Seq2Seq(encoder=encoder, decoder=decoder)
        print(model)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_function = nn.CrossEntropyLoss(ignore_index=self.input_ix.objs_to_ints[common_conf.PAD_TOKEN])

        for epoch in range(0, epochs):

            random.shuffle(self.train_data)
            x_padded, y_padded = get_xy(data=self.train_data,
                                        enc_pad_ix=self.input_ix.objs_to_ints[common_conf.PAD_TOKEN],
                                        dec_pad_ix=self.output_ix.objs_to_ints[common_conf.PAD_TOKEN])

            epoch_loss = 0
            for start_ix in range(0, len(self.train_data), batch_size):
                x = get_batch(data=x_padded, start_ix=start_ix, batch_size=batch_size)
                y = get_batch(data=y_padded, start_ix=start_ix, batch_size=batch_size)

                optimizer.zero_grad()
                y_pred = model(x=x, y=y, teacher_forcing=parser_conf.teacher_force_train)
                y_pred = y_pred.view(-1, y_pred.shape[-1])
                y_true = y.view(-1)

                loss = loss_function(y_pred, y_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                epoch_loss += loss.item()
            print(epoch_loss/len(self.train_data))
            print('current dev acc')
            pred_deriv = self.decode(test_data=self.dev_data, model=model)
            evaluate(dev_data=self.dev_data, pred_derivations=pred_deriv)
            if (epoch+1) % 5 == 0:
                lr = parser_conf.initial_lr
            else:
                lr = lr/2

        return model

    def decode(self, test_data: List[Example], model=None) -> List[List[Derivation]]:
        # Implement the inference here
        test_derivs = []
        for data_point in test_data:
            y_tok = []
            x, y = get_xy([data_point],
                          enc_pad_ix=self.input_ix.objs_to_ints[common_conf.PAD_TOKEN],
                          dec_pad_ix=self.output_ix.objs_to_ints[common_conf.PAD_TOKEN])
            if model is not None:
                y_pred = model(x=x, y=y, teacher_forcing=parser_conf.teacher_force_test)
            else:
                y_pred = self.model(x=x, y=y, teacher_forcing=parser_conf.teacher_force_test)
            y_pred = y_pred.view(-1, y_pred.shape[-1])
            for t in y_pred:
                pred_ix_t = t.argmax(0).item()
                tok_t = self.output_ix.ints_to_objs[pred_ix_t]
                y_tok.append(tok_t)

            test_derivs.append([Derivation(data_point, 0.3, y_tok)])

        return test_derivs
        # raise Exception("implement me!")
