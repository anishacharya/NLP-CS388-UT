from common.utils.embedding import WordEmbedding
import common.common_config as common_conf

import torch
import torch.nn as nn
import numpy as np


class RNNEncoder(nn.Module):
    def __init__(self, conf, word_embed: WordEmbedding):
        super(RNNEncoder, self).__init__()
        self.conf = conf
        self.word_embed = word_embed
        self.embed_weight_init = torch.from_numpy(np.asarray(list(word_embed.ix2embed.values())))
        # self.nb_classes = self.conf.no_classes
        self.hidden_size_rnn = self.conf.hidden_size
        # self.rec_unit = conf.no_of_rec_unit
        self.nb_rec_units = conf.no_of_rec_units
        self.rnn_dropout = conf.rnn_dropout
        self.dropout = conf.dropout

        self.embedding = nn.Embedding(len(self.word_embed.word_ix),
                                      embedding_dim=self.word_embed.emb_dim,
                                      padding_idx=self.word_embed.word_ix.add_and_get_index(common_conf.PAD_TOKEN))
        # _weight=self.embed_weight_init)
        self.rnn = nn.LSTM(input_size=self.word_embed.emb_dim,
                           hidden_size=self.hidden_size_rnn,
                           num_layers=self.nb_rec_units,
                           bidirectional=False,
                           dropout=self.rnn_dropout,
                           batch_first=True)
        # self.hidden2tag = nn.Linear(in_features=self.hidden_size_rnn * 2,  # *2 since Bidirectional
        #                             out_features=self.nb_classes)
        self.dropout = nn.Dropout(p=self.dropout)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.copy_(self.embed_weight_init)
        # nn.init.xavier_uniform_(self.hidden2tag.weight)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, data_batch):
        embedded_data = self.embedding(data_batch)

        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        rnn_out, (rnn_hidden, rnn_cell) = self.rnn(embedded_data)
        # concat from both the directions
        # rnn_hidden = self.dropout(torch.cat((rnn_hidden[-2, :, :], rnn_hidden[-1, :, :]), dim=1))
        # posterior = torch.sigmoid(self.hidden2tag(rnn_hidden).squeeze(1))

        return rnn_hidden, rnn_cell


class RNNDecoder(nn.Module):
    def __init__(self, conf, word_embed: WordEmbedding):
        super(RNNDecoder, self).__init__()
        self.conf = conf
        self.word_embed = word_embed
        self.embed_weight_init = torch.from_numpy(np.asarray(list(word_embed.ix2embed.values())))
        # self.nb_classes = self.conf.no_classes
        self.hidden_size_rnn = self.conf.hidden_size
        # self.rec_unit = conf.no_of_rec_unit
        self.nb_rec_units = conf.no_of_rec_units
        self.rnn_dropout = conf.rnn_dropout
        self.dropout = conf.dropout

        self.embedding = nn.Embedding(len(self.word_embed.word_ix),
                                      embedding_dim=self.word_embed.emb_dim,
                                      padding_idx=self.word_embed.word_ix.add_and_get_index(common_conf.PAD_TOKEN))
        # _weight=self.embed_weight_init)
        self.rnn = nn.LSTM(input_size=self.word_embed.emb_dim,
                           hidden_size=self.hidden_size_rnn,
                           num_layers=self.nb_rec_units,
                           bidirectional=False,
                           dropout=self.rnn_dropout,
                           batch_first=True)
        # self.hidden2tag = nn.Linear(in_features=self.hidden_size_rnn * 2,  # *2 since Bidirectional
        #                             out_features=self.nb_classes)
        self.dropout = nn.Dropout(p=self.dropout)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.copy_(self.embed_weight_init)
        # nn.init.xavier_uniform_(self.hidden2tag.weight)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)


class RNNSeq2Seq(nn.Module):
    def __init__(self, encoder: RNNEncoder, decoder: RNNDecoder):
        super(RNNSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        rnn_cell, rnn_hidden = self.encoder(x)




