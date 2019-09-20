from random import shuffle
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import time

from src.data_utils.definitions import PersonExample
from src.feature_extractors.utils import create_index, index_data, load_word_embedding
from src.feature_extractors.embedding_features import word_embedding
from src.utils.utils import Indexer, flatten
import src.config as config

glove_file = '/Users/anishacharya/Desktop/glove.6B/glove.6B.300d.txt'


class BinaryPersonClassifier(object):
    """
    Classifier to classify a token in a sentence as a PERSON token or not.
    Constructor arguments are merely suggestions; you're free to change these.
    """

    def __init__(self, model, word_ix: Indexer, pos_ix: Indexer, ix2embed: Dict):
        self.model = model
        self.word_ix = word_ix
        self.pos_ix = pos_ix
        self.ix2embed = ix2embed

    def predict(self, tokens: List[str], idx: int):
        """
        Makes a prediction for token at position idx in the given PersonExample
        Returns 0 if not a person token, 1 if a person token
        """
        token = tokens[idx]

        "Word Indicator"
        # word_indicator_feat_dim = len(self.word_ix)
        # word_indicator = [0] * word_indicator_feat_dim
        # if token.word.lower() in self.word_ix.objs_to_ints:
        #     ix = self.word_ix.objs_to_ints[token.word.lower()]
        #     word_indicator[ix] = 1
        # else:
        #     word_indicator[self.word_ix.objs_to_ints['__UNK__']] = 1

        "pos indicator"
        # pos_indicator_feat_dim = len(self.pos_ix)
        # pos_indicator = [0] * pos_indicator_feat_dim
        # if token.pos in self.pos_ix.objs_to_ints:
        #     ix = self.pos_ix.objs_to_ints[token.pos]
        #     pos_indicator[ix] = 1
        # else:
        #     pos_indicator[self.pos_ix.objs_to_ints['__UNK__']] = 1
        #
        "starts with a capital"
        # is_upper = is_upper_indicator_feat(word=token.word, idx=idx)
        # # all caps
        # is_all_caps = all_caps_indicator_feat(word=token.word)

        " Current word Embedding "
        if token.word.lower() in self.word_ix.objs_to_ints:
            token_ix = self.word_ix.objs_to_ints[token.word.lower()]
        else:
            token_ix = self.word_ix.objs_to_ints['__UNK__']
        if token_ix in self.ix2embed:
            word_emb = self.ix2embed[token_ix]
        else:
            word_emb = self.ix2embed[self.word_ix.objs_to_ints['__UNK__']]

        # Context vector
        # sentence = [token.word for token in tokens]
        # context_window_1 = get_context_vector(tokens=sentence,
        #                                       idx=idx,
        #                                       window_len=1,
        #                                       word2ix=self.word_ix.objs_to_ints,
        #                                       ix2embed=self.ix2embed)

        # context_window_2 = get_context_vector(tokens=sentence,
        #                                       idx=idx,
        #                                       window_len=2,
        #                                       word2ix=self.word_ix.objs_to_ints,
        #                                       ix2embed=self.ix2embed)
        # context_left_1 = get_context_vector(tokens=sentence,
        #                                     idx=idx,
        #                                     window_len=1,
        #                                     word2ix=self.word_ix.objs_to_ints,
        #                                     ix2embed=self.ix2embed,
        #                                     left=True)
        # context_left_2 = get_context_vector(tokens=sentence,
        #                                     idx=idx,
        #                                     window_len=2,
        #                                     word2ix=self.word_ix.objs_to_ints,
        #                                     ix2embed=self.ix2embed,
        #                                     left=True)
        # context_right_1 = get_context_vector(tokens=sentence,
        #                                     idx=idx,
        #                                     window_len=1,
        #                                     word2ix=self.word_ix.objs_to_ints,
        #                                     ix2embed=self.ix2embed,
        #                                     right=True)
        feat_vec = []
        # feat_vec = feat_vec + word_indicator
        # feat_vec = feat_vec + pos_indicator
        # feat_vec = feat_vec + is_upper
        # feat_vec = feat_vec + is_all_caps
        #
        feat_vec = feat_vec + word_emb
        # feat_vec = feat_vec + context_window_1
        # feat_vec = feat_vec + context_window_2
        # feat_vec = feat_vec + context_left_1
        # feat_vec = feat_vec + context_left_2
        # feat_vec = feat_vec + context_right_1

        x_test = torch.FloatTensor(feat_vec)
        y_hat_test = self.model(x_test)

        if y_hat_test[0] > y_hat_test[1]:
            return 0
        return 1


def get_features(sentence: List,
                 pos: List,
                 word_ix: Indexer,
                 pos_ix: Indexer,
                 embed_ix: Dict,
                 idx: int) -> List:
    word = word_ix.ints_to_objs[sentence[idx]]

    "collect 1-hot / Indicator Features"
    # word_indicator = word_indicator_feat(sentence=sentence, word_ix=word_ix, idx=idx)
    # pos_indicator = pos_indicator_feat(pos=pos, pos_ix=pos_ix, idx=idx)
    # is_upper_indicator = is_upper_indicator_feat(word=word, idx=idx)
    # all_caps_indicator = all_caps_indicator_feat(word=word)

    "collect word embedding features"
    word_embed = word_embedding(word=word,
                                ix2embed=embed_ix,
                                word2ix=word_ix.objs_to_ints)

    "get context window __ | __ embedding (average)"
    # tokens = inverse_idx_sentence(sentence, ix2word=word_ix.ints_to_objs)
    # context_window_1 = get_context_vector(tokens=tokens,
    #                                       idx=idx,
    #                                       window_len=1,
    #                                       word2ix=word_ix.objs_to_ints,
    #                                       ix2embed=embed_ix)
    # context_window_2 = get_context_vector(tokens=tokens,
    #                                       idx=idx,
    #                                       window_len=2,
    #                                       word2ix=word_ix.objs_to_ints,
    #                                       ix2embed=embed_ix)
    # context_left_1 = get_context_vector(tokens=tokens,
    #                                     idx=idx,
    #                                     window_len=1,
    #                                     word2ix=word_ix.objs_to_ints,
    #                                     ix2embed=embed_ix,
    #                                     left=True)
    # context_left_2 = get_context_vector(tokens=tokens,
    #                                     idx=idx,
    #                                     window_len=2,
    #                                     word2ix=word_ix.objs_to_ints,
    #                                     ix2embed=embed_ix,
    #                                     left=True)
    # context_right_1 = get_context_vector(tokens=tokens,
    #                                     idx=idx,
    #                                     window_len=1,
    #                                     word2ix=word_ix.objs_to_ints,
    #                                     ix2embed=embed_ix,
    #                                     right=True)

    # gather all features
    feature = []

    # feature = feature + word_indicator
    # feature = feature + pos_indicator
    # feature = feature + is_upper_indicator
    # feature = feature + all_caps_indicator
    # #
    feature = feature + word_embed
    # feature = feature + context_window_1
    # # feature = feature + context_window_2
    # feature = feature + context_left_1
    # # feature = feature + context_left_2
    # feature = feature + context_right_1
    return feature


def train_model_based_binary_ner(ner_exs: List[PersonExample]):
    shuffle(ner_exs)
    """
    =======================================
    ========== Build Indexers =============
    """
    word_ix, pos_ix = create_index(ner_exs)
    ix2embedding = load_word_embedding(pretrained_embedding_filename=glove_file,
                                       word2index_vocab=word_ix.objs_to_ints)
    train_sent, POS, train_lables = index_data(ner_exs, word_ix, pos_ix)

    epochs = config.epochs
    batch_size = config.batch_size
    initial_lr = config.initial_lr
    no_of_classes = config.no_of_classes

    """
    ==================================
    =====  Network Definition ========
    ==================================
    """
    word_indicator_feat_dim = len(word_ix)
    pos_indicator_feat_dim = len(pos_ix)
    is_upper_feat_dim = 1
    all_caps_indicator_feat_dim = 1

    word_embedding_feat_dim = 300
    context_window_1 = 300
    context_window_2 = 300
    context_left_1 = 300
    context_left_2 = 300
    context_right_1 = 300

    feat_dim = 0

    # feat_dim += word_indicator_feat_dim
    # feat_dim += pos_indicator_feat_dim
    # feat_dim += is_upper_feat_dim
    # feat_dim += all_caps_indicator_feat_dim
    #
    feat_dim += word_embedding_feat_dim
    # feat_dim += context_window_1
    # feat_dim += context_window_2
    # feat_dim += context_left_1
    # # feat_dim += context_left_2
    # feat_dim += context_right_1

    n_input_dim = feat_dim
    n_hidden1 = 16  # Number of hidden nodes
    n_hidden2 = 8
    n_output = 2  # Number of output nodes = for binary classifier

    net = nn.Sequential(
        nn.Linear(n_input_dim, n_hidden1),
        nn.ELU(),
        nn.Linear(n_hidden1, n_hidden2),
        nn.ELU(),
        nn.Linear(n_hidden2, n_output),
        nn.Sigmoid())
    print(net)

    learning_rate = initial_lr

    for epoch in range(epochs):
        t = time.time()
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

            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

            loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # loss_func = nn.BCELoss()

            Y_train = np.asarray(Y_train)
            X_train = []
            for sent, pos in zip(data_batch, pos_batch):
                for idx in range(0, len(sent)):
                    X_train.append(get_features(sent, pos, word_ix, pos_ix, ix2embedding, idx))
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

        print("Epoch : ", epoch)
        print("Time taken", time.time() - t)
        print("Learning Rate = ", learning_rate)
        if (epoch + 1) % 3 == 0:
            learning_rate = initial_lr*2

        learning_rate = learning_rate / 2
        print("----------")
        print(" ")

    return BinaryPersonClassifier(model=net,
                                  word_ix=word_ix,
                                  pos_ix=pos_ix,
                                  ix2embed=ix2embedding)


