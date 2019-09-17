from src.data_utils.definitions import LabeledSentence, chunks_from_bio_tag_seq, Token
from typing import List
from collections import Counter


class LabelCountNER(object):
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


def train_label_count_ner(training_set: List[LabeledSentence]) -> LabelCountNER:
    """
    :param training_set: labeled NER sentences to extract a BadNerModel from
    :return: the BadNerModel based on counts collected from the training data
    """
    words_to_tag_counters = {}
    for sentence in training_set:
        tags = sentence.get_bio_tags()
        for idx in range(0, len(sentence)):
            word = sentence.tokens[idx].word
            if word not in words_to_tag_counters:
                words_to_tag_counters[word] = Counter()
            words_to_tag_counters[word][tags[idx]] += 1.0
    return LabelCountNER(words_to_tag_counters)
