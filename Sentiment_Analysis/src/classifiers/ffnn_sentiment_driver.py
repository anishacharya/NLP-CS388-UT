from Sentiment_Analysis.src.data_utils.definitions import SentimentExample
from Sentiment_Analysis.src.utils import get_xy
import Sentiment_Analysis.sentiment_config as sentiment_conf

from common.utils.embedding import WordEmbedding, SentenceEmbedding
import common.common_config as common_conf
from common.utils.utils import get_batch
from common.models.FFNN import FFNN

from typing import List
import numpy as np
from random import shuffle
import torch
import torch.optim as optim


def train_sentiment_ffnn(train_data: List[SentimentExample],
                         dev_data: List[SentimentExample],
                         word_embed: WordEmbedding) -> FFNN:

    # define training components
    model = FFNN(300, 150, 2)
    lr = sentiment_conf.initial_lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = sentiment_conf.ffnn_epochs
    batch_size = sentiment_conf.batch_size

    for epoch in range(0, epochs):
        # shuffle data
        shuffle(train_data)
        total_loss = 0.0
        for start_ix in range(0, len(train_data), batch_size):
            train_batch = get_batch(data=train_data, start_ix=start_ix, batch_size=batch_size)
            x_batch, y_batch = get_xy(data=train_batch, word_embed=word_embed)
            model.zero_grad()
            probs = model.forward(x_batch)
            # Can also use built-in NLLLoss as a shortcut here (takes log probabilities) but we're being explicit here
            loss = torch.neg(torch.log(probs)).dot(y_batch)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Loss on epoch %i: %f" % (epoch, total_loss))
    return model
