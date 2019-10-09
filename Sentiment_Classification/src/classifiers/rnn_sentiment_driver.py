from common.utils.embedding import WordEmbedding
import common.common_config as common_conf
from common.models.RNN import RNN
from common.utils.utils import get_batch

from Sentiment_Classification.src.data_utils.definitions import SentimentExample
import Sentiment_Classification.sentiment_config as sentiment_conf
from Sentiment_Classification.src.utils import get_xy_padded

import torch
import torch.optim as optim
import torch.nn as nn

from typing import List
# from random import shuffle
from sklearn.utils import shuffle


def train_sentiment_rnn(train_data: List[SentimentExample],
                        dev_data: List[SentimentExample],
                        word_embed: WordEmbedding):
    rec_unit = sentiment_conf.rec_unit
    vocab_size = len(word_embed.word_ix)
    model = RNN(conf=sentiment_conf,
                vocab_size=vocab_size,
                emb_dim=word_embed.emb_dim,
                weights_init=word_embed.ix2embed,)

    acc = 0.0
    lr = sentiment_conf.initial_lr

    epochs = sentiment_conf.epochs
    batch_size = sentiment_conf.batch_size
#    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.BCELoss()

    x_padded, y_padded = get_xy_padded(data=train_data, word_embed=word_embed)

    for epoch in range(0, epochs):
        # shuffle(train_data)
        # shuffle(x_padded, y_padded)
        total_loss = 0.0

        for start_ix in range(0, len(train_data), batch_size):
            x_batch = get_batch(data=x_padded, start_ix=start_ix, batch_size=batch_size)
            y_batch = get_batch(data=y_padded, start_ix=start_ix, batch_size=batch_size)

