import numpy as np
from data_utils.nerdata import PersonExample
from utils.utils import Indexer
from joblib import Parallel, delayed
import multiprocessing

import torch
"""
Author: Anish Acharya <anishacharya@utexas.edu>
"""


class NERFeatureExtractors:
    def __init__(self,
                 word_index: Indexer,
                 pos_index: Indexer):

        np.random.seed(1)

        self.word_ix = word_index
        self.pos_ix = pos_index

        self.feature_dim = len(word_index)  # + len(pos_index)

    def uni_gram_index_feature(self,
                               ner_ex: PersonExample) -> torch.tensor:
        """
        :return: a |V| dim vector A per word with A[i] = 1 if index(wi) = i, the corresponding label
        so if there are n tokens return x = n * feature_dim , Y = n
        """
        # x = np.zeros((len(ner_ex), self.feature_dim))
        x = torch.zeros([len(ner_ex), self.feature_dim])

        for idx in range(0, len(ner_ex)):
            token = ner_ex.tokens[idx]
            x[idx, :] = self.uni_gram_index_feature_per_token(token)
        return x

    def uni_gram_index_feature_per_token(self, token):
        x = torch.zeros([1, self.feature_dim])
        if token.word not in self.word_ix.objs_to_ints:
            j = self.word_ix.objs_to_ints['__UNK__']
        else:
            j = self.word_ix.objs_to_ints[token.word]
        x[:, j] = 1
        return x

    def pos_index_feature(self,
                          ner_ex: PersonExample):
        """
        :return: if n unique POS in the training data then returns a n dim vector B, the corresponding label
        for wi where B[i]=1 if index(POS(wi)) = i
        """
        raise NotImplementedError
