
"""
Author: Anish Acharya <anishacharya@utexas.edu>
"""


def word_indicator_feat(sentence, idx, word_ix):
    word_indicator_feat_dim = len(word_ix)
    word_indicator = [0] * word_indicator_feat_dim
    word_indicator[sentence[idx]] = 1
    return word_indicator


def pos_indicator_feat(pos, pos_ix, idx):
    pos_indicator_feat_dim = len(pos_ix)
    pos_indicator = [0] * pos_indicator_feat_dim
    pos_indicator[pos[idx]] = 1
    return pos_indicator


def is_upper_indicator_feat(word, idx):
    # check if word starts with capital
    is_upper_indicator = [0]
    if idx != 0 and word[0].isupper:
        is_upper_indicator[0] = 1
    return is_upper_indicator


def all_caps_indicator_feat(word):
    is_all_caps = [0]
    if word.isupper:
        is_all_caps[0] = 1
    return is_all_caps



