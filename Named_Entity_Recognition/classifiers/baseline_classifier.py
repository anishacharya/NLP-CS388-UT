from evaluation.ner_eval import *
from models.logistic_regression import *
from utils.utils import Indexer
from data_utils.nerdata import PersonExample, Token
from feature_extractors.ner_features import NERFeatureExtractors



from typing import List
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

"""
Author: Anish Acharya <anishacharya@utexas.edu>
Adopted From: Greg Durret <gdurrett@cs.utexas.edu>
"""


class BaselineClassifier(object):
    """
    Classifier to classify a token in a sentence as a PERSON token or not.
    Constructor arguments are merely suggestions; you're free to change these.
    """

    def __init__(self,
                 data: List[PersonExample],
                 dev_data: List[PersonExample]):
        # Loop through each example and build index, and inverse index

        shuffle(data)
        self.word_ix, self.pos_ix = create_index(data)
        self.model = self.train(data=data, dev_data=dev_data)
        print("done")

    def train(self, data, dev_data) -> LogisticRegression:
        # TODO: create and move to config
        lr = 0.001
        epochs = 1
        batch_size = 64
        print_iter = 10

        no_of_batches = len(data) // batch_size
        feature_dim = len(self.word_ix)
        out_feature_dim = 2

        criterion = nn.NLLLoss()
        model = LogisticRegression(input_size=feature_dim, num_classes=out_feature_dim)
        print(model)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            t = time.time()
            curr_batch = 0
            running_loss = 0
            for i in range(no_of_batches):
                next_batch = (curr_batch + batch_size)
                data_batch = data[curr_batch: next_batch]
                x_train_batch, y_train_batch = project_to_continuous_space(ner_exs=data_batch,
                                                                           word_ix=self.word_ix,
                                                                           pos_ix=self.pos_ix)
                curr_batch = next_batch
                optimizer.zero_grad()

                y_pred_batch = model(x_train_batch)
                loss = criterion(y_pred_batch, y_train_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if i % print_iter == 0:
                    print("===== Current Loss ====", running_loss/(i+1))
                running_loss = 0
            print("==== Time Taken for this epoch===", time.time() - t)
        return model

    def predict(self,
                tokens: List[Token],
                idx: int):
        """
        Makes a prediction for token at position idx in the given PersonExample
        :param tokens:
        :param idx:
        :return: 0 if not a person token, 1 if a person token
        """
        token = tokens[idx]
        # instance of the feature extractor
        projector = NERFeatureExtractors(word_index=self.word_ix,
                                         pos_index=self.pos_ix)
        # Convert to feature space
        x_test = projector.uni_gram_index_feature_per_token(token=token)
        output = self.model(x_test)
        _, predicted = torch.max(output.data, 1)
        return predicted


def create_index(ner_exs: List[PersonExample]) -> [Indexer, Indexer]:
    word_ix = Indexer()
    pos_ix = Indexer()
    for ex in ner_exs:
        for idx in range(0, len(ex)):
            word_ix.add_and_get_index(ex.tokens[idx].word)
            pos_ix.add_and_get_index(ex.tokens[idx].pos)

    # create index for unseen objects
    word_ix.add_and_get_index('__UNK__')
    pos_ix.add_and_get_index('__UNK__')
    return word_ix, pos_ix


def project_to_continuous_space(ner_exs: List[PersonExample],
                                word_ix: Indexer,
                                pos_ix: Indexer):
    # instance of the feature extractor
    projector = NERFeatureExtractors(word_index=word_ix,
                                     pos_index=pos_ix)
    # x_train = np.empty((0, projector.feature_dim))
    x_train = torch.zeros(0, projector.feature_dim)
    y_train = []
    for ner_ex in ner_exs:
        y = ner_ex.labels
        x = projector.uni_gram_index_feature(ner_ex=ner_ex)
        x_train = torch.cat((x_train, x), 0)
        y_train = np.concatenate((y_train, y))

        # x_train = np.concatenate((x_train, x), axis=0)
    # y_train = y_train.reshape(len(y_train), 1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    return x_train, y_train


def run_model_based_binary_ner(data: List[PersonExample],
                               dev_data: List[PersonExample]) -> BaselineClassifier:
    return BaselineClassifier(data=data, dev_data=dev_data)
