from utils.utils import Indexer
from data_utils.nerdata import PersonExample, Token
from feature_extractors.ner_features import NERFeatureExtractors

from typing import List

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
                 data,
                 word_index: Indexer):
                 #pos_index: Indexer):

        self.data = data  # usually training data in continuous space
        self.word_indexer = word_index
        #self.pos_indexer = pos_index

        # project data into feature space

    def predict(self,
                tokens: List[Token],
                idx: int):
        """
        Makes a prediction for token at position idx in the given PersonExample
        :param tokens:
        :param idx:
        :return: 0 if not a person token, 1 if a person token
        """
        raise Exception("Implement me!")


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

# def project_to_continuous_space():


def run_model_based_binary_ner(ner_exs: List[PersonExample]) -> BaselineClassifier:
    # Loop through each example and build index, and inverse index
    word_ix, pos_ix = create_index(ner_exs)

    return BaselineClassifier(ner_exs, word_ix)



