import numpy as np
from data_utils.nerdata import PersonExample
from utils.utils import Indexer

from typing import List


class NERFeatureExtractors:
    def __init__(self,
                 word_index: Indexer,
                 pos_index: Indexer,
                 ner_exs: List[PersonExample]):

        np.random.seed(1)

        self.word_ix = word_index
        self.pos_ix = pos_index
        self.data = ner_exs

    def uni_gram_index_feature(self):
        """
        :return: a |V| dim vector A per word with A[i] = 1 if index(wi) = i
        """

    def pos_index_feature(self):
        """
        :return: if n unique POS in the training data then returns a n dim vector B
        for wi where B[i]=1 if index(POS(wi)) = i
        """


