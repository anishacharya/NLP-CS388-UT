from src.data_utils.definitions import (LabeledSentence,
                                        chunks_from_bio_tag_seq,
                                        Token)
from typing import List, Counter


class LabelCountBinary(object):
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


class LabelCount(object):
    """
    NER model that simply assigns each word its most likely observed tag in training

    Attributes:
        words_to_tag_counters: dictionary where each word (string) is mapped to a Counter over tags representing
        counts observed in training
    """

    def __init__(self, words_to_tag_counters):
        self.words_to_tag_counters = words_to_tag_counters

    def decode(self, sentence_tokens: List[Token]) -> LabeledSentence:
        """
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        pred_tags = []
        for tok in sentence_tokens:
            if tok.word in self.words_to_tag_counters:
                # [0] selects the top most common (tag, count) pair, the next [0] picks out the tag itself
                pred_tags.append(self.words_to_tag_counters[tok.word].most_common(1)[0][0])
            else:
                pred_tags.append("O")
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))
