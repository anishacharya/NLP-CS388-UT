from src.data_utils.definitions import LabeledSentence, Indexer
from src.data_utils.utils import get_word_index

from src.models.hmm import HmmNerModel

from typing import List
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
from string import punctuation


stops = set(stopwords.words("english"))
stops.update(set(punctuation))
stops.update({'-X-', ',', '$', ':', '-DOCSTART-'})


def train_hmm_ner(sentences: List[LabeledSentence]) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("__UNK__")
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

    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
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

    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    print(repr(init_counts))
    init_counts = np.log(init_counts / init_counts.sum())  # P(s[0] = tag[i])
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])  # P(tag[j]|tag[i])
    # similar to transitions
    emission_counts = (emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])  # P(t(i)|W(j))
    print("Tag indexer: %s" % tag_indexer)
    print("Initial state log probabilities: %s" % init_counts)
    print("Transition log probabilities: %s" % transition_counts)
    print("Emission log probs too big to print...")
    print("Emission log probs for India: %s" % emission_counts[:, word_indexer.add_and_get_index("India")])
    print("Emission log probs for Phil: %s" % emission_counts[:, word_indexer.add_and_get_index("Phil")])
    print("note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)
