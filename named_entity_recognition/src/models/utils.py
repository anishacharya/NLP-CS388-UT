from typing import List
import numpy as np
from src.data_utils.definitions import Token, Indexer, LabeledSentence
import torch
import src.config as conf


class ProbabilisticSequenceScorer(object):
    """
    Scoring function for sequence networks based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs: np.ndarray,
                 transition_log_probs: np.ndarray, emission_log_probs: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) \
            else self.word_indexer.index_of("__UNK__")
        return self.emission_log_probs[tag_idx, word_idx]


def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)


def load_word_embedding(pretrained_embedding_filename, word2index_vocab):
    """
    Read a GloVe txt file.we return dictionaries
    `mapping index to embedding vector( index_to_embedding)`,
    """
    index_to_embedding = {}
    with open(pretrained_embedding_filename, 'r') as glove_file:
        for (i, line) in enumerate(glove_file):
            split = line.split(' ')
            word = split[0]
            ix = word2index_vocab.get(word.lower())
            if ix is not None:
                representation = split[1:]
                representation = np.array([float(val) for val in representation])
                index_to_embedding[ix] = list(representation)
    unk = word2index_vocab[conf.UNK_TOKEN]
    index_to_embedding[unk] = [0.0] * len(representation)  # Empty representation for unknown words.

    return index_to_embedding


def sigmoid(z):
    return 1 / (1 + np.e ** (-z))


def logistic_regression_loss_imbalanced(y_true, y_hat, alpha=1):
    t1 = alpha * y_true * np.log(y_hat + 10e-5)
    t2 = (1 - y_true) * np.log(1 - y_hat + 10e-5)

    return -np.sum(t1 + t2)


def maybe_add_feature(feats: List[int], feature_indexer: Indexer, add_to_indexer: bool, feat: str):
    """
    :param feats: list[int] features that we've built so far
    :param feature_indexer: Indexer object to apply
    :param add_to_indexer: True if we should expand the Indexer, false otherwise. If false, we discard feat if it isn't
    in the indexer
    :param feat: new feature to index and potentially add
    :return:
    """
    if add_to_indexer:
        feats.append(feature_indexer.add_and_get_index(feat))
    else:
        feat_idx = feature_indexer.index_of(feat)
        if feat_idx != -1:
            feats.append(feat_idx)


def score_indexed_features(feats, weights: np.ndarray):
    """
    Computes the dot product over a list of features (i.e., a sparse feature vector)
    and a weight vector (numpy array)
    :param feats: List[int] or numpy array of int features
    :param weights: numpy array
    :return: the score
    """
    score = 0.0
    for feat in feats:
        score += weights[feat]
    return score


def prepare_data_point(sentence: LabeledSentence, word_indexer: Indexer):
    one_hot = [word_indexer.objs_to_ints[token.word] if token.word in word_indexer.objs_to_ints
               else word_indexer.objs_to_ints[conf.UNK_TOKEN] for token in sentence.tokens]
    return torch.tensor(one_hot, dtype=torch.long)


def prepare_label_point(sentence: LabeledSentence, tag_indexer: Indexer):
    one_hot = [tag_indexer.objs_to_ints[bio_tag] if bio_tag in tag_indexer.objs_to_ints else
               tag_indexer.objs_to_ints[conf.UNK_TOKEN] for bio_tag in sentence.bio_tags]
    return torch.tensor(one_hot, dtype=torch.long)


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def get_triangular_lr(iteration, stepsize, base_lr, max_lr):
    """Given the inputs, calculates the lr that should be applicable for this iteration"""
    """http://teleported.in/posts/cyclic-learning-rate/"""
    bump = float(max_lr - base_lr)/float(stepsize)
    cycle = iteration%(2*stepsize)
    if cycle < stepsize:
        lr = base_lr + cycle*bump
    else:
        lr = max_lr - (cycle-stepsize)*bump
    return lr



