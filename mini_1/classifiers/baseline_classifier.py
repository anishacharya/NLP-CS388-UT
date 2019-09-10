from utils import *
from optimizers import *
from data_utils.nerdata import *

"""
Author: Anish Acharya <anishacharya@utexas.edu>
Adopted From: Greg Durret <gdurrett@cs.utexas.edu>
"""


class PersonClassifier(object):
    """
    Classifier to classify a token in a sentence as a PERSON token or not.
    Constructor arguments are merely suggestions; you're free to change these.
    """

    def __init__(self, weights: np.ndarray, indexer: Indexer):
        self.weights = weights
        self.indexer = indexer

    def predict(self, tokens: List[Token], idx: int):
        """
        Makes a prediction for token at position idx in the given PersonExample
        :param tokens:
        :param idx:
        :return: 0 if not a person token, 1 if a person token
        """
        raise Exception("Implement me!")


def train_base_classifier(ner_exs: List[PersonExample]):
    raise Exception("Implement me!")