from collections import Counter
from typing import List

from src.data_utils.definitions import LabeledSentence, PersonExample
from src.models.label_count import LabelCount, LabelCountBinary


def train_label_count_binary_ner(ner_exs: List[PersonExample]) -> LabelCountBinary:
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
    return LabelCountBinary(pos_counts, neg_counts)


def train_label_count_ner(training_set: List[LabeledSentence]) -> LabelCount:
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
    return LabelCount(words_to_tag_counters)
