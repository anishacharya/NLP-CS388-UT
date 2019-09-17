"""
Author: Anish Acharya <anishacharya@utexas.edu>
Adopted From: Greg Durret <gdurrett@cs.utexas.edu>
"""

from typing import List
from src.data_utils.definitions import (LabeledSentence,
                                        PersonExample,
                                        Token,
                                        bio_tags_from_chunks,
                                        chunks_from_bio_tag_seq)


def read_data(file: str) -> List[LabeledSentence]:
    """
    Reads a data-set in the CoNLL format from a file
    The format is one token per line:
    [word] [POS] [syntactic chunk] *potential junk column* [NER tag]
    One blank line appears after each sentence
    :param file: string filename to read
    :return: list[LabeledSentence]
    """
    f = open(file)
    sentences = []
    curr_tokens = []
    curr_bio_tags = []
    for line in f:
        stripped = line.strip()
        if stripped != "":
            fields = stripped.split(" ")
            if len(fields) == 4 or len(fields) == 5:
                curr_tokens.append(Token(fields[0], fields[1], fields[2]))
                curr_bio_tags.append(fields[-1])
        elif stripped == "" and len(curr_tokens) > 0:
            sentences.append(LabeledSentence(curr_tokens, chunks_from_bio_tag_seq(curr_bio_tags)))
            curr_tokens = []
            curr_bio_tags = []
    return sentences


def transform_label_for_binary_classification(ner_exs: List[LabeledSentence], pos_class:str = 'PER'):
    """
    :param pos_class: which class to treat as Positive : default Person class
    :param ner_exs: List of chunk-style NER examples
    :return: A list of PersonExamples extracted from the NER data
    """
    for labeled_sent in ner_exs:
        tags = bio_tags_from_chunks(labeled_sent.chunks, len(labeled_sent))
        labels = [1 if tag.endswith(pos_class) else 0 for tag in tags]

        yield PersonExample(labeled_sent.tokens,
                            labels)
