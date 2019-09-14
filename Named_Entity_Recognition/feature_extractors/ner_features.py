
"""
Author: Anish Acharya <anishacharya@utexas.edu>
"""


def word_indicator():
    raise NotImplementedError


def is_upper_indicator_feat(word, idx):
    # check if word starts with capital
    is_upper_indicator = [0]
    if idx != 0 and word[0].isupper:
        is_upper_indicator[0] = 1
    return is_upper_indicator



