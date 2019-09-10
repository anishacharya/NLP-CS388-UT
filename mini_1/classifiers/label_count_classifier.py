from data_utils.nerdata import *
from collections import Counter

"""
Author: Anish Acharya <anishacharya@utexas.edu>
Adopted From: Greg Durret <gdurrett@cs.utexas.edu>
"""


class CountBasedPersonClassifier(object):
    """
    Person classifier that takes counts of how often a word was observed to be the positive and negative class
    in training, and classifies as positive any tokens which are observed to be positive more than negative.
    Unknown tokens or ties default to negative.
    Attributes:
        pos_counts: how often each token occurred with the label 1 in training
        neg_counts: how often each token occurred with the label 0 in training
    """
    def __init__(self, pos_counts: Counter, neg_counts: Counter):
        self.pos_counts = pos_counts
        self.neg_counts = neg_counts

    def predict(self, tokens: List[Token], idx: int):
        if self.pos_counts[tokens[idx].word] > self.neg_counts[tokens[idx].word]:
            return 1
        else:
            return 0


def run_count_based_binary_ner(ner_exs: List[PersonExample]) -> CountBasedPersonClassifier:
    """
    :param ner_exs: training examples to build the count-based classifier from
    :return: A CountBasedPersonClassifier using counts collected from the given examples
    """
    pos_counts = Counter()
    neg_counts = Counter()
    for ex in ner_exs:
        for idx in range(0, len(ex)):
            if ex.labels[idx] == 1:
                pos_counts[ex.tokens[idx].word] += 1.0
            else:
                neg_counts[ex.tokens[idx].word] += 1.0
    return CountBasedPersonClassifier(pos_counts, neg_counts)
