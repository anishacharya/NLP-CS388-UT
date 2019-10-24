from src.data_utils.definitions import LabeledSentence, Indexer
from src.data_utils.utils import get_word_index


from src.models.hmm import HmmNerModel
import src.config as conf

from typing import List
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
from string import punctuation


# stops = set(stopwords.words("english"))
# stops = set(punctuation)
# stops.update(set(punctuation))
# stops.update({'-X-', ',', '$', ':', '-DOCSTART-'})
stops = set()


def train_hmm_ner(sentences: List[LabeledSentence]) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index(conf.UNK_TOKEN)
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer=word_indexer, word_counter=word_counter, stops=stops, word=token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)

    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer), len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer), len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.add_and_get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer=word_indexer, word_counter=word_counter,
                                      stops=stops, word=sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                transition_counts[tag_indexer.add_and_get_index(bio_tags[i-1])][tag_idx] += 1.0

    init_counts = np.log(init_counts / init_counts.sum())  # P(s[0] = tag[i])

    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])  # P(tag[j]|tag[i])
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])  # P(W(j)|t(i))
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)
