from common.utils.embedding import WordEmbedding
import common.common_config as common_conf
from common.models.RNN import RNN
from common.utils.utils import get_batch

from Sentiment_Classification.src.data_utils.definitions import SentimentExample
import Sentiment_Classification.sentiment_config as sentiment_conf
from Sentiment_Classification.src.utils import get_xy_padded, get_xy
from Sentiment_Classification.src.evaluation.evaluate import evaluate_sentiment, evaluate_sentiment_simple
from Sentiment_Classification.src.data_utils.rotten_tomatoes_reader import write_sentiment_examples

import torch
import torch.optim as optim
import torch.nn as nn

from typing import List
# from random import shuffle
from sklearn.utils import shuffle


def train_sentiment_rnn(train_data: List[SentimentExample],
                        dev_data: List[SentimentExample],
                        test_data: List[SentimentExample],
                        word_embed: WordEmbedding):
    model = RNN(conf=sentiment_conf, word_embed=word_embed)
    acc = 0.0
    last_epoch_acc = 0.0
    lr = sentiment_conf.initial_lr
    lr_decay = sentiment_conf.lr_decay
    weight_decay = sentiment_conf.weight_decay

    epochs = sentiment_conf.epochs
    batch_size = sentiment_conf.batch_size
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)
    loss_function = nn.BCELoss()

    x_padded, y_padded = get_xy_padded(data=train_data, word_embed=word_embed)
    # x, y = get_xy(data=train_data)

    for epoch in range(0, epochs):
        # if (epoch + 1) % 5 == 0:
        #     lr = sentiment_conf.initial_lr
        print('Learning Rate is: {}'.format(lr))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # shuffle(train_data)
        # shuffle(x_padded, y_padded)
        total_loss = 0.0
        no_of_batch = 0
        for start_ix in range(0, len(train_data), batch_size):
            x_batch = get_batch(data=x_padded, start_ix=start_ix, batch_size=batch_size)
            y_batch = get_batch(data=y_padded, start_ix=start_ix, batch_size=batch_size)

            probs = model(x_batch)
            loss = loss_function(probs, y_batch)
            total_loss += loss / batch_size
            no_of_batch += 1
            model.zero_grad()
            loss.backward()
            optimizer.step()
            # print('Current Batch Loss is: {}'.format(loss/batch_size))

        print("Loss on epoch %i: %f" % (epoch, total_loss/no_of_batch))
        # _, metrics = evaluate_sentiment(model=model, data=dev_data,
        #                                 word_embedding=word_embed, model_type='RNN')
        _, accuracy = evaluate_sentiment_simple(model=model, data=dev_data,
                                                word_embedding=word_embed, model_type='RNN')
        print(" ========  Performance after epoch {} is ====== ".format(epoch))
        print("New Accuracy = ", accuracy)

        if accuracy > acc:
            acc = accuracy
            print("==== saving model ====")
            torch.save(model.state_dict(), sentiment_conf.model_path)
            y_pred, _ = evaluate_sentiment_simple(model=model, model_type='RNN',
                                                  word_embedding=word_embed, data=test_data)
            test_predicted = []
            for pred, data_point in zip(y_pred, test_data):
                test_predicted.append(SentimentExample(label=int(pred), indexed_words=data_point.indexed_words))
            # Write the test set output
            print('writing Test Output')
            write_sentiment_examples(test_predicted, sentiment_conf.output_path, word_embed.word_ix)
            print('Done Writing Test Output')
            lr = lr/2
        elif (accuracy - last_epoch_acc) < - 0.01:
            lr = sentiment_conf.initial_lr/2
        else:
            lr = lr/2
        last_epoch_acc = accuracy

