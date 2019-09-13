import numpy as np
from data_utils.nerdata import PersonExample
from utils.utils import Indexer, flatten
from nltk.corpus import stopwords
from string import punctuation
from typing import List
"""
Author: Anish Acharya <anishacharya@utexas.edu>
"""


def create_index(ner_exs: List[PersonExample]) -> [Indexer, Indexer]:
    stops = set(stopwords.words("english"))
    stops.update(set(punctuation))
    stops.update({'-X-', ',', '$', ':', '-DOCSTART-'})
    # stops = set()
    word_ix = Indexer()
    pos_ix = Indexer()

    # create index for unseen objects
    word_ix.add_and_get_index('__UNK__')
    pos_ix.add_and_get_index('__UNK__')

    for ex in ner_exs:
        for idx in range(0, len(ex)):
            token = ex.tokens[idx].word
            pos = ex.tokens[idx].pos
            if token not in stops:
                word_ix.add_and_get_index(token)
            if pos not in stops:
                pos_ix.add_and_get_index(pos)

    return word_ix, pos_ix


class Feature:
    def __init__(self, feature, size):
        self.feature = feature
        self.size = size


def sparse_feature_encoder(ner_exs: List[PersonExample],
                           word_ix: Indexer,
                           pos_ix: Indexer):
    projector = NERFeatureExtractors(word_index=word_ix,
                                     pos_index=pos_ix)

    x_train_uni = []
    x_train_pos = []

    y_train = []
    for ner_ex in ner_exs:
        y = ner_ex.labels
        x1 = projector.get_uni_gram_one_hot_compressed(ner_ex=ner_ex) ## Unigram index 1-hot
        x2 = projector.get_pos_one_hot_compressed(ner_ex=ner_ex) ## POS 1-hot

        x_train_uni.append(x1)
        x_train_pos.append(x2)

        y_train = np.concatenate((y_train, y))

    # wrap in Feature class combine all features into one list to return
    features = [Feature(x_train_uni, len(word_ix)), Feature(x_train_pos, len(pos_ix))]
    return features, y_train


def sparse_feature_decoder(encoded_feature):
    feat_dim = 0
    for feat in encoded_feature:
        feat_dim += feat.size
    token_count = len(flatten(encoded_feature[0].feature))
    decoded_feat = np.empty((token_count, feat_dim))

    # also append the ix of non-zero terms for sparse gradient updates
    non_zero_grad_loc = [[] for i in range(token_count)]

    start_ix = 0
    for feat in encoded_feature:
        token_position = 0  # start from first data point i.e. token
        curr_feat = feat.feature  # should be list of sentences encoded : [w1 w2][w2 w9 w189][w11 w13] ...
        for sent in curr_feat:
            for token in sent:
                decoded_feat[token_position, start_ix + token] = 1
                non_zero_grad_loc[token_position].append(token)
                token_position += 1
        start_ix += feat.size
    return decoded_feat, non_zero_grad_loc


# def project_to_continuous_space(ner_exs: List[PersonExample],
#                                 word_ix: Indexer,
#                                 pos_ix: Indexer):
#     # instance of the feature extractor
#     projector = NERFeatureExtractors(word_index=word_ix,
#                                      pos_index=pos_ix)
#     x_train = np.empty((0, projector.feature_dim))
#     y_train = []
#     for ner_ex in ner_exs:
#         y = ner_ex.labels
#         x1 = projector.uni_gram_index_feature(ner_ex=ner_ex)
#         x2 = projector.pos_index_feature(ner_ex=ner_ex)
#
#         x_train = np.concatenate((x_train, x1), axis=0)
#         x_train = np.concatenate((x_train, x2), axis=0)
#
#         y_train = np.concatenate((y_train, y))
#
#     return x_train, y_train


class NERFeatureExtractors:
    def __init__(self,
                 word_index: Indexer,
                 pos_index: Indexer):

        self.word_ix = word_index
        self.pos_ix = pos_index

        self.unigram_feat_dim = len(word_index)
        self.pos_feat_dim = len(pos_index)

        self.feature_dim = self.unigram_feat_dim + self.pos_feat_dim

    def get_uni_gram_one_hot_compressed(self,
                                        ner_ex: PersonExample):
        x = []
        for idx in range(0, len(ner_ex)):
            token = ner_ex.tokens[idx]
            if token.word not in self.word_ix.objs_to_ints:
                j = self.word_ix.objs_to_ints['__UNK__']
            else:
                j = self.word_ix.objs_to_ints[token.word]
            x.append(j)
        return x

    def get_pos_one_hot_compressed(self,
                                   ner_ex: PersonExample):
        x = []
        for idx in range(0, len(ner_ex)):
            token = ner_ex.tokens[idx]
            if token.pos not in self.pos_ix.objs_to_ints:
                j = self.pos_ix.objs_to_ints['__UNK__']
            else:
                j = self.pos_ix.objs_to_ints[token.pos]
            x.append(j)
        return x

    # def uni_gram_index_feature(self,
    #                            ner_ex: PersonExample):
    #     """
    #     :return: a |V| dim vector A per word with A[i] = 1 if index(wi) = i, the corresponding label
    #     so if there are n tokens return x = n * feature_dim , Y = n
    #     """
    #     x = np.zeros((len(ner_ex), self.unigram_feat_dim))
    #
    #     for idx in range(0, len(ner_ex)):
    #         token = ner_ex.tokens[idx]
    #         x[idx, :] = self.uni_gram_index_feature_per_token(token)
    #     return x

    def uni_gram_index_feature_per_token(self, token):
        x = np.zeros([1, self.unigram_feat_dim])
        if token.word not in self.word_ix.objs_to_ints:
            j = self.word_ix.objs_to_ints['__UNK__']
        else:
            j = self.word_ix.objs_to_ints[token.word]
        x[:, j] = 1
        return x

    # def pos_index_feature(self,
    #                       ner_ex: PersonExample):
    #     """
    #     :return: if n unique POS in the training data then returns a n dim vector B, the corresponding label
    #     for wi where B[i]=1 if index(POS(wi)) = i
    #     """
    #     x = np.zeros((len(ner_ex), self.unigram_feat_dim))
    #     for idx in range(0, len(ner_ex)):
    #         token = ner_ex.tokens[idx]
    #         x[idx, :] = self.pos_index_feature_per_token(token)
    #     return x

    def pos_index_feature_per_token(self, token):
        x = np.zeros([1, self.pos_feat_dim])
        if token.pos not in self.pos_ix.objs_to_ints:
            j = self.pos_ix.objs_to_ints['__UNK__']
        else:
            j = self.pos_ix.objs_to_ints[token.pos]
        x[:, j] = 1
        return x




