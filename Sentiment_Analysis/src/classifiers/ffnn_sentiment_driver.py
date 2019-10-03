from Sentiment_Analysis.src.data_utils.definitions import SentimentExample
from Sentiment_Analysis.src.utils import get_xy_FFNN
import Sentiment_Analysis.sentiment_config as sentiment_conf
from Sentiment_Analysis.src.evaluation.evaluate import evaluate_sentiment

from common.utils.embedding import WordEmbedding, SentenceEmbedding
from common.utils.utils import get_batch
from common.models.FFNN import FFNN

from typing import List
from random import shuffle

import torch.optim as optim
import torch.nn as nn


def train_sentiment_ffnn(train_data: List[SentimentExample],
                         dev_data: List[SentimentExample],
                         word_embed: WordEmbedding) -> FFNN:

    # define training components
    model = FFNN(300, 150, 2)
    best_model = model
    acc = 0
    lr = sentiment_conf.initial_lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = sentiment_conf.ffnn_epochs
    batch_size = sentiment_conf.batch_size
    loss_function = nn.BCELoss()

    for epoch in range(0, epochs):
        # shuffle data
        shuffle(train_data)
        total_loss = 0.0
        for start_ix in range(0, len(train_data), batch_size):
            train_batch = get_batch(data=train_data, start_ix=start_ix, batch_size=batch_size)
            x_batch, y_batch = get_xy_FFNN(data=train_batch, word_embed=word_embed)
            model.zero_grad()
            probs = model(x_batch)
            loss = loss_function(probs, y_batch)
            total_loss += loss / batch_size
            loss.backward()
            optimizer.step()
        print("Loss on epoch %i: %f" % (epoch, total_loss))
        _, metrics = evaluate_sentiment(model=model, data=dev_data,
                                        word_embedding=word_embed, model_type='FFNN')
        # to make sure no deep/shallow copy issue:
        _, metrics_best = evaluate_sentiment(model=best_model, data=dev_data,
                                             word_embedding=word_embed, model_type='FFNN')
        print(" ========  Performance after epoch {} is ====== ".format(epoch))
        print("New Accuracy = ", metrics.accuracy)
        print("Current Best Model Acc: ", metrics_best.accuracy)
        if metrics.accuracy > acc:
            best_model = model




    return best_model
