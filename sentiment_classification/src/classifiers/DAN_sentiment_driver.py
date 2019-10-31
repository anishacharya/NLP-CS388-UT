from common.utils.embedding import WordEmbedding
import common.common_config as common_conf
from common.networks.RNN import RNN
from common.utils.utils import get_batch

from sentiment_classification.src.data_utils.definitions import SentimentExample
import sentiment_classification.sentiment_config as sentiment_conf
from sentiment_classification.src.utils import get_xy_padded, get_xy
from sentiment_classification.src.evaluation.evaluate import evaluate_sentiment

import torch
import torch.optim as optim
import torch.nn as nn

from typing import List
# from random import shuffle
from sklearn.utils import shuffle


def train_sentiment_rnn(train_data: List[SentimentExample],
                        dev_data: List[SentimentExample],
                        word_embed: WordEmbedding):
    model = RNN(conf=sentiment_conf, word_embed=word_embed)
    acc = 0.0
    lr = sentiment_conf.initial_lr

    epochs = sentiment_conf.epochs
    batch_size = sentiment_conf.batch_size
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.BCELoss()

    x_padded, y_padded = get_xy_padded(data=train_data, word_embed=word_embed)
    # x, y = get_xy(data=train_data)

    for epoch in range(0, epochs):
        # shuffle(train_data)
        # shuffle(x_padded, y_padded)
        total_loss = 0.0
        no_of_batch = 0
        for start_ix in range(0, len(train_data), batch_size):
            x_batch = get_batch(data=x_padded, start_ix=start_ix, batch_size=batch_size)
            y_batch = get_batch(data=y_padded, start_ix=start_ix, batch_size=batch_size)
            model.zero_grad()
            probs = model(x_batch)
            loss = loss_function(probs, y_batch)
            total_loss += loss / batch_size
            no_of_batch += 1
            loss.backward()
            optimizer.step()
            # print('Current Batch Loss is: {}'.format(loss/batch_size))

        print("Loss on epoch %i: %f" % (epoch, total_loss/no_of_batch))
        _, metrics = evaluate_sentiment(model=model, data=dev_data,
                                        word_embedding=word_embed, model_type='RNN')
        print(" ========  Performance after epoch {} is ====== ".format(epoch))
        print("New Accuracy = ", metrics.accuracy)
        if metrics.accuracy > acc:
            acc = metrics.accuracy
            print("==== saving model ====")
            torch.save(model.state_dict(), sentiment_conf.model_path)

