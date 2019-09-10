import numpy as np
from data_utils.nerdata import PersonExample
from utils.utils import Indexer

from typing import List

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

        self.feature_dim = len(word_index) + len(pos_index) + 2 # 1 for unk_ word one for unk_ pos


    def uni_gram_index_feature(self,
                               ner_ex: PersonExample):
        """
        :return: a |V| dim vector A per word with A[i] = 1 if index(wi) = i, the corresponding label
        so if there are n tokens return X = n * feature_dim , Y = n
        """
        X = np.zeros(len(ner_ex), self.feature_dim)
        Y = np.asarray(ner_ex.labels)

        # for idx in range(0, len(ner_ex)):




    def pos_index_feature(self,
                          ner_ex: PersonExample):
        """
        :return: if n unique POS in the training data then returns a n dim vector B, the corresponding label
        for wi where B[i]=1 if index(POS(wi)) = i
        """


