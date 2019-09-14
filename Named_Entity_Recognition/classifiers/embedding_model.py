from random import shuffle
from typing import List

import numpy as np
import torch
import torch.nn as nn

from data_utils.nerdata import PersonExample
from feature_extractors.utils import create_index
from utils.utils import Indexer, flatten


glove_file = '/Users/anishacharya/Desktop/PhD_1_1/NLP/CS388/Named_Entity_Recognition/data/glove.6B/glove.6B.300d.txt'


class BaselineNERClassifier(object):
    """
    Classifier to classify a token in a sentence as a PERSON token or not.
    Constructor arguments are merely suggestions; you're free to change these.
    """

    def __init__(self, model, word_ix: Indexer, pos_ix: Indexer):
        self.model = model
        self.word_ix = word_ix
        self.pos_ix = pos_ix

    def predict(self, tokens: List[str], idx: int):
        """
        Makes a prediction for token at position idx in the given PersonExample
        :param tokens:
        :param idx:
        :return: 0 if not a person token, 1 if a person token
        """
        token = tokens[idx]

        word_indicator_feat_dim = len(self.word_ix)
        word_indicator = [0] * word_indicator_feat_dim
        if token.word in self.word_ix.objs_to_ints:
            ix = self.word_ix.objs_to_ints[token.word]
            word_indicator[ix] = 1
        else:
            word_indicator[self.word_ix.objs_to_ints['__UNK__']] = 1

        pos_indicator_feat_dim = len(self.pos_ix)
        pos_indicator = [0] * pos_indicator_feat_dim
        if token.pos in self.pos_ix.objs_to_ints:
            ix = self.pos_ix.objs_to_ints[token.pos]
            pos_indicator[ix] = 1
        else:
            pos_indicator[self.pos_ix.objs_to_ints['__UNK__']] = 1

        is_upper = [0]
        if idx != 0 and token.word[0].isupper:
            is_upper[0] = 1

        feat_vec = []
        feat_vec = feat_vec + word_indicator
        feat_vec = feat_vec + pos_indicator
        feat_vec = feat_vec + is_upper

        x_test = torch.FloatTensor(feat_vec)
        y_hat_test = self.model(x_test)

        if y_hat_test[0] > y_hat_test[1]:
            return 0
        return 1


def get_features(sentence, pos, word_ix, pos_ix, idx):
    # collect 1-hot of word
    word_indicator_feat_dim = len(word_ix)
    word_indicator = [0] * word_indicator_feat_dim
    word_indicator[sentence[idx]] = 1

    # collect POS
    pos_indicator_feat_dim = len(pos_ix)
    pos_indicator = [0] * pos_indicator_feat_dim
    pos_indicator[pos[idx]] = 1

    # check if word starts with capital
    is_upper_indicator = [0]
    word = word_ix.ints_to_objs[sentence[idx]]
    if idx !=0 and word[0].isupper:
        is_upper_indicator[0] = 1

    # collect embedding

    # gather all features
    feature = []
    feature = feature + word_indicator
    feature = feature + pos_indicator
    feature = feature + is_upper_indicator

    return feature


def train_model_based_ner(ner_exs: List[PersonExample], dev_data):
    shuffle(ner_exs)
    word_ix, pos_ix = create_index(ner_exs)

    train_sent, POS, train_lables = index_data(ner_exs, word_ix, pos_ix)

    epochs = 15
    batch_size = 124
    print_iter = 15
    no_of_classes = 2

    """
    ===============================================
    =====  Network Definition ===================== 
    """
    word_indicator_feat_dim = len(word_ix)
    pos_indicator_feat_dim = len(pos_ix)
    is_upper_feat_dim = 1

    feat_dim = 0
    feat_dim += word_indicator_feat_dim
    feat_dim += pos_indicator_feat_dim
    feat_dim += is_upper_feat_dim

    n_input_dim = feat_dim
    n_hidden = 4  # Number of hidden nodes
    n_output = 2  # Number of output nodes = for binary classifier

    net = nn.Sequential(
        nn.Linear(n_input_dim, n_hidden),
        nn.ELU(),
        nn.Linear(n_hidden, n_output),
        nn.Sigmoid())

    print(net)

    learning_rate = 0.01
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        """
        ================= Create batch ===============
        """
        for i in range(0, len(train_sent), batch_size):
            if len(train_sent[i:]) <= batch_size:
                data_batch = train_sent[i:]
                pos_batch = POS[i:]
            else:
                data_batch = train_sent[i: i + batch_size]
                pos_batch = POS[i: i + batch_size]

            Y_train = flatten(train_lables[i: i + batch_size])
            if i/100 % print_iter == 0:
                print('processing batch = ', i/100)

            """
            ========== scaling ================== 
            """
            # compute class weights
            if Y_train.count(1) == 0:
                scaling = 1
            else:
                scaling = Y_train.count(0) / Y_train.count(1)

            pos_weight = torch.ones([no_of_classes])
            pos_weight[1] = scaling

            loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            Y_train = np.asarray(Y_train)

            X_train = []
            for sent, pos in zip(data_batch, pos_batch):
                for idx in range(0, len(sent)):
                    X_train.append(get_features(sent, pos, word_ix, pos_ix, idx))
            X_train = np.asarray(X_train)

            # One hot
            y_train_one_hot = np.zeros((Y_train.size, no_of_classes))
            for ix, n in enumerate(Y_train):
                if n == 0:
                    y_train_one_hot[ix, 0] = 1
                else:
                    y_train_one_hot[ix, 1] = 1
            Y_train = y_train_one_hot

            # convert to tensor
            X_train_t = torch.FloatTensor(X_train)
            Y_train_t = torch.FloatTensor(Y_train)

            y_hat = net(X_train_t)

            loss = loss_func(y_hat, Y_train_t)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i/100 % print_iter == 0:
                print('Loss ==> ', loss.item())

        evaluate_classifier(dev_data, BaselineNERClassifier(model=net, word_ix=word_ix, pos_ix=pos_ix))
        learning_rate = learning_rate/2
    return BaselineNERClassifier(model=net, word_ix=word_ix, pos_ix=pos_ix)


def index_data(ner_exs: List[PersonExample], word_ix, pos_ix):
    # convert data to index
    Sentences = []
    POS = []
    indexed_y = []

    for sent in ner_exs:
        s = []
        pos = []
        indexed_y.append(sent.labels)
        for token in sent.tokens:
            if token.word in word_ix.objs_to_ints:
                s.append(word_ix.objs_to_ints[token.word])
            else:
                s.append(word_ix.objs_to_ints['__UNK__'])
            if token.pos in pos_ix.objs_to_ints:
                pos.append(pos_ix.objs_to_ints[token.pos])
            else:
                pos.append(pos_ix.objs_to_ints['__UNK__'])
        Sentences.append(s)
        POS.append(pos)
    return Sentences, POS, indexed_y


def evaluate_classifier(exs: List[PersonExample], classifier: BaselineNERClassifier):
    """
    Prints evaluation of the classifier on the given examples
    :param exs: PersonExample instances to run on
    :param classifier: classifier to evaluate
    """
    predictions = []
    golds = []
    for ex in exs:
        for idx in range(0, len(ex)):
            golds.append(ex.labels[idx])
            predictions.append(classifier.predict(ex.tokens, idx))
    print_evaluation(golds, predictions)


def print_evaluation(golds: List[int], predictions: List[int]):
    """
    Prints statistics about accuracy, precision, recall, and F1
    :param golds: list of {0, 1}-valued ground-truth labels for each token in the test set
    :param predictions: list of {0, 1}-valued predictions for each token
    :return:
    """
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    if len(golds) != len(predictions):
        raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
    for idx in range(0, len(golds)):
        gold = golds[idx]
        prediction = predictions[idx]
        if prediction == gold:
            num_correct += 1
        if prediction == 1:
            num_pred += 1
        if gold == 1:
            num_gold += 1
        if prediction == 1 and gold == 1:
            num_pos_correct += 1
        num_total += 1
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
    print("Precision: %i / %i = %f" % (num_pos_correct, num_pred, prec))
    print("Recall: %i / %i = %f" % (num_pos_correct, num_gold, rec))
    print("F1: %f" % f1)
