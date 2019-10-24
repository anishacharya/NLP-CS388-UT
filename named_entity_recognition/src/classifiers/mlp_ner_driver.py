import time
from collections import Counter
from random import shuffle
from string import punctuation
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from nltk.corpus import stopwords

import src.config as conf
from src.data_utils.definitions import Indexer, LabeledSentence
from src.data_utils.utils import get_word_index
from src.feature_extractors.embedding_features import get_word_embedding, get_context_vector
from src.models.utils import load_word_embedding
from src.utils.utils import flatten
from src.models.mlp import MLPNerClassifier
from src.evaluation.ner_eval import write_test_output, print_evaluation_metric
from src.data_utils.utils import inverse_idx_sentence
from src.feature_extractors.indicator_features import pos_indicator_feat,is_upper_indicator_feat,all_caps_indicator_feat
from src.models.utils import get_triangular_lr

stops = set()
stops.update(stopwords.words("english"))
stops.update(set(punctuation))
stops.update({'-X-', ',', '$', ':', '-DOCSTART-'})


def get_features(sentence: List,
                 pos: List,
                 word_ix: Indexer,
                 pos_ix: Indexer,
                 embed_ix: Dict,
                 idx: int) -> List:
    word = word_ix.ints_to_objs[sentence[idx]]

    "collect 1-hot / Indicator Features"
    # word_indicator = word_indicator_feat(sentence=sentence, word_ix=word_ix, idx=idx)
    pos_indicator = pos_indicator_feat(pos=pos, pos_ix=pos_ix, idx=idx)
    is_upper_indicator = is_upper_indicator_feat(word=word, idx=idx)
    all_caps_indicator = all_caps_indicator_feat(word=word)

    "collect word embedding features"
    word_embed = get_word_embedding(word=word,
                                    ix2embed=embed_ix,
                                    word2ix=word_ix.objs_to_ints)

    "get context window __ | __ embedding (average)"
    tokens = inverse_idx_sentence(sentence, ix2word=word_ix.ints_to_objs)
    context_window_1 = get_context_vector(tokens=tokens,
                                          idx=idx,
                                          window_len=1,
                                          word2ix=word_ix.objs_to_ints,
                                          ix2embed=embed_ix)
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
    feature = feature + pos_indicator
    feature = feature + is_upper_indicator
    feature = feature + all_caps_indicator
    # #
    feature = feature + word_embed
    feature = feature + context_window_1
    # # feature = feature + context_window_2
    # feature = feature + context_left_1
    # # feature = feature + context_left_2
    # feature = feature + context_right_1
    return feature


def train_mlp_ner(train_data: List[LabeledSentence], dev_data, test_data):
    shuffle(train_data)
    """
    =======================================
    ========== Build Indexers =============
    """
    tag_ix = Indexer()
    word_ix = Indexer()
    pos_ix = Indexer()
    word_counter = Counter()

    tag_ix.add_and_get_index(conf.PAD_TOKEN)   # padding
    word_ix.add_and_get_index(conf.PAD_TOKEN)
    tag_ix.add_and_get_index(conf.EOS_TOKEN)   # End of Sentence
    word_ix.add_and_get_index(conf.EOS_TOKEN)
    tag_ix.add_and_get_index(conf.BOS_TOKEN)   # Beginning of Sentence
    word_ix.add_and_get_index(conf.BOS_TOKEN)
    tag_ix.add_and_get_index(conf.UNK_TOKEN)   # Unk Words
    word_ix.add_and_get_index(conf.UNK_TOKEN)

    for sentence in train_data:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0

    for sentence in train_data:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer=word_ix, word_counter=word_counter, stops=stops, word=token.word.lower(), th=0)
            pos_ix.add_and_get_index(token.pos)
        for tag in sentence.get_bio_tags():
            tag_ix.add_and_get_index(tag)

    ix2embedding = load_word_embedding(pretrained_embedding_filename=conf.glove_file,
                                       word2index_vocab=word_ix.objs_to_ints)
    train_sent = []
    POS = []
    train_labels = []

    for sentence in train_data:
        s = []
        pos = []
        labels = []
        for token in sentence.tokens:
            if token.word.lower() in word_ix.objs_to_ints:
                s.append(word_ix.objs_to_ints[token.word.lower()])
            else:
                s.append(word_ix.objs_to_ints[conf.UNK_TOKEN])

            if token.pos in pos_ix.objs_to_ints:
                pos.append(pos_ix.objs_to_ints[token.pos])
            else:
                pos.append(pos_ix.objs_to_ints[conf.UNK_TOKEN])

        for tag in sentence.get_bio_tags():
            labels.append(tag_ix.objs_to_ints[tag])
        train_sent.append(s)
        POS.append(pos)
        train_labels.append(labels)

    epochs = conf.epochs
    batch_size = conf.batch_size
    initial_lr = conf.initial_lr
    no_of_classes = len(tag_ix)

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
    feat_dim += pos_indicator_feat_dim
    feat_dim += is_upper_feat_dim
    feat_dim += all_caps_indicator_feat_dim
    #
    feat_dim += word_embedding_feat_dim
    feat_dim += context_window_1
    # feat_dim += context_window_2
    # feat_dim += context_left_1
    # # feat_dim += context_left_2
    # feat_dim += context_right_1

    n_input_dim = feat_dim
    n_hidden1 = 64  # Number of hidden nodes
    n_hidden2 = 32
    n_hidden3 = 16
    n_output = no_of_classes  # Number of output nodes = for binary classifier

    net = nn.Sequential(
        nn.Linear(n_input_dim, n_hidden1),
        nn.ELU(),
        nn.Dropout(0.2),
        nn.Linear(n_hidden1, n_hidden2),
        nn.ELU(),
        nn.Dropout(0.2),
        nn.Linear(n_hidden2, n_hidden3),
        nn.ELU(),
        nn.Linear(n_hidden3, n_output),
        nn.Sigmoid())
    print(net)

    learning_rate = initial_lr

    best_f1 = 0
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

            Y_train = flatten(train_labels[i: i + batch_size])
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
            loss_func = nn.BCELoss()
            Y_train = np.asarray(Y_train)
            X_train = []
            for sent, pos in zip(data_batch, pos_batch):
                for idx in range(0, len(sent)):
                    X_train.append(get_features(sent, pos, word_ix, pos_ix, ix2embedding, idx))
            X_train = np.asarray(X_train)

            # One hot
            y_train_one_hot = np.zeros((Y_train.size, no_of_classes))

            for ix, n in enumerate(Y_train):
                y_train_one_hot[ix, n] = 1
            Y_train = y_train_one_hot

            # convert to tensor
            X_train_t = torch.FloatTensor(X_train)
            Y_train_t = torch.FloatTensor(Y_train)

            y_hat = net(X_train_t)

            loss = loss_func(y_hat, Y_train_t)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model = MLPNerClassifier(model=net,
                                 word_ix=word_ix,
                                 pos_ix=pos_ix,
                                 tag_ix=tag_ix,
                                 ix2embed=ix2embedding)
        # Compute Dev Acc.
        # if (epoch + 1) % 3 == 0:
        dev_decoded = [model.decode(test_ex.tokens) for test_ex in dev_data]
        f1 = print_evaluation_metric(dev_data, dev_decoded)
        if f1 > best_f1:
            test_decoded = [model.decode(test_ex.tokens) for test_ex in test_data]
            write_test_output(test_decoded, conf.output_path)

        print("-------------------------")
        print("Epoch: ", epoch)
        print("Time taken: ", time.time() - t)
        print(" ")
        print(" -------------------------")
        print("----------")
        print(" ")

    return model



