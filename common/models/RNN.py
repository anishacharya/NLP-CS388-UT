import torch
import torch.nn as nn
import numpy as np
from common.utils.embedding import WordEmbedding
from common.common_config import PAD_TOKEN


class RNN(nn.Module):
    def __init__(self, conf, word_embed: WordEmbedding):
        super(RNN, self).__init__()
        self.conf = conf
        self.word_embed = word_embed
        self.embed_weight_init = torch.from_numpy(np.asarray(list(word_embed.ix2embed.values())))
        self.nb_classes = self.conf.no_classes
        self.hidden_size_rnn = self.conf.hidden_size
        self.rec_unit = conf.rec_unit
        self.nb_rec_units = conf.no_of_rec_units
        self.rnn_dropout = conf.rnn_dropout
        self.dropout = conf.dropout

        self.embedding = nn.Embedding(len(self.word_embed.word_ix),
                                      embedding_dim=self.word_embed.emb_dim,
                                      padding_idx=self.word_embed.word_ix.add_and_get_index(PAD_TOKEN))
        # _weight=self.embed_weight_init)
        self.rnn = nn.LSTM(input_size=self.word_embed.emb_dim,
                           hidden_size=self.hidden_size_rnn,
                           num_layers=self.nb_rec_units,
                           bidirectional=True,
                           dropout=self.rnn_dropout)
        self.hidden2tag = nn.Linear(in_features=self.hidden_size_rnn * 2,  # *2 since Bidirectional
                                    out_features=self.nb_classes)
        self.dropout = nn.Dropout(p=self.dropout)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.copy_(self.embed_weight_init)
        nn.init.xavier_uniform(self.hidden2tag.weight)
        nn.init.xavier_uniform(self.rnn.all_weights)

    def forward(self, data_batch):
        raise NotImplementedError
