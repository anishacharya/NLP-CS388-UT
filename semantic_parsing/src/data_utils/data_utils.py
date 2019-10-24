from common.utils.indexer import Indexer
from typing import List, Tuple
import random
from collections import Counter


class Example(object):
    """
    Wrapper class for a single (natural language, logical form) input/output (x/y) pair
    Attributes:
        x: the natural language as one string
        x_tok: tokenized natural language as a list of strings
        x_indexed: indexed tokens, a list of ints
        y: the raw logical form as a string
        y_tok: tokenized logical form, a list of strings
        y_indexed: indexed logical form, a list of ints
    """
    def __init__(self, x: str, x_tok: List[str], x_indexed: List[int], y, y_tok, y_indexed):
        self.x = x
        self.x_tok = x_tok
        self.x_indexed = x_indexed
        self.y = y
        self.y_tok = y_tok
        self.y_indexed = y_indexed

    def __repr__(self):
        return " ".join(self.x_tok) + " => " + " ".join(self.y_tok) + "\n   indexed as: " + repr(self.x_indexed) + " => " + repr(self.y_indexed)

    def __str__(self):
        return self.__repr__()


#
class Derivation(object):
    """
    Wrapper for a possible solution returned by the model associated with an Example. Note that y_toks here is a
    predicted y_toks, and the Example itself contains the gold y_toks.
    Attributes:
          example: The underlying Example we're predicting on
          p: the probability associated with this prediction
          y_toks: the tokenized output prediction
    """
    def __init__(self, example: Example, p, y_toks):
        self.example = example
        self.p = p
        self.y_toks = y_toks

    def __str__(self):
        return "%s (%s)" % (self.y_toks, self.p)

    def __repr__(self):
        return self.__str__()


PAD_SYMBOL = "<PAD>"
UNK_SYMBOL = "<UNK>"
SOS_SYMBOL = "<SOS>"
EOS_SYMBOL = "<EOS>"


def load_datasets(train_path: str, dev_path: str, test_path: str, domain=None) -> (List[Tuple[str,str]], List[Tuple[str,str]], List[Tuple[str,str]]):
    """
    Reads the training, dev, and test data from the corresponding files.
    :param train_path:
    :param dev_path:
    :param test_path:
    :param domain: Ignore this parameter
    :return:
    """
    train_raw = load_dataset(train_path, domain=domain)
    dev_raw = load_dataset(dev_path, domain=domain)
    test_raw = load_dataset(test_path, domain=domain)
    return train_raw, dev_raw, test_raw


def load_dataset(filename: str, domain="geo") -> List[Tuple[str,str]]:
    """
    Reads a dataset in from the given file.
    :param filename:
    :param domain: Ignore this parameter
    :return: a list of untokenized, unindexed (natural language, logical form) pairs
    """
    dataset = []
    num_pos = 0
    with open(filename) as f:
        for line in f:
            x, y = line.rstrip('\n').split('\t')
            # Geoquery features some additional preprocessing of the logical form
            if domain == "geo":
                y = geoquery_preprocess_lf(y)
            dataset.append((x, y))
    print("%i / %i pos exs" % (num_pos, len(dataset)))
    return dataset


def tokenize(x) -> List[str]:
    """
    :param x: string to tokenize
    :return: x tokenized with whitespace tokenization
    """
    return x.split()


def index(x_tok: List[str], indexer: Indexer) -> List[int]:
    return [indexer.index_of(xi) if indexer.index_of(xi) >= 0 else indexer.index_of(UNK_SYMBOL) for xi in x_tok]


def index_data(data, input_indexer: Indexer, output_indexer: Indexer, example_len_limit):
    """
    Indexes the given data
    :param data:
    :param input_indexer:
    :param output_indexer:
    :param example_len_limit:
    :return:
    """
    data_indexed = []
    for (x, y) in data:
        x_tok = tokenize(x)
        y_tok = tokenize(y)[0:example_len_limit]
        data_indexed.append(Example(x, x_tok, index(x_tok, input_indexer), y, y_tok,
                                          index(y_tok, output_indexer) + [output_indexer.index_of(EOS_SYMBOL)]))
    return data_indexed


def index_datasets(train_data, dev_data, test_data, example_len_limit, unk_threshold=0.0) -> (List[Example], List[Example], List[Example], Indexer, Indexer):
    """
    Indexes train and test datasets where all words occurring less than or equal to unk_threshold times are
    replaced by UNK tokens.
    :param train_data:
    :param dev_data:
    :param test_data:
    :param example_len_limit:
    :param unk_threshold: threshold below which words are replaced with unks. If 0.0, the model doesn't see any
    UNKs at train time
    :return:
    """
    input_word_counts = Counter()
    # Count words and build the indexers
    for (x, y) in train_data:
        for word in tokenize(x):
            input_word_counts[word] += 1.0
    input_indexer = Indexer()
    output_indexer = Indexer()
    # Reserve 0 for the pad symbol for convenience
    input_indexer.add_and_get_index(PAD_SYMBOL)
    input_indexer.add_and_get_index(UNK_SYMBOL)
    output_indexer.add_and_get_index(PAD_SYMBOL)
    output_indexer.add_and_get_index(SOS_SYMBOL)
    output_indexer.add_and_get_index(EOS_SYMBOL)
    # Index all input words above the UNK threshold
    for word in input_word_counts.keys():
        if input_word_counts[word] > unk_threshold + 0.5:
            input_indexer.add_and_get_index(word)
    # Index all output tokens in train
    for (x, y) in train_data:
        for y_tok in tokenize(y):
            output_indexer.add_and_get_index(y_tok)
    # Index things
    train_data_indexed = index_data(train_data, input_indexer, output_indexer, example_len_limit)
    dev_data_indexed = index_data(dev_data, input_indexer, output_indexer, example_len_limit)
    test_data_indexed = index_data(test_data, input_indexer, output_indexer, example_len_limit)
    return train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer


##################################################
# YOU SHOULD NOT NEED TO LOOK AT THESE FUNCTIONS #
##################################################
def print_evaluation_results(test_data, selected_derivs, denotation_correct, example_freq=50, print_output=True):
    """
    Prints output and accuracy. YOU SHOULD NOT NEED TO CALL THIS DIRECTLY -- instead call evaluate in main.py, which
    wraps this.
    :param test_data:
    :param selected_derivs:
    :param denotation_correct:
    :param example_freq: How often to print output
    :param print_output: True if we should print the scores, false otherwise (you should never need to set this False)
    :return:
    """
    num_exact_match = 0
    num_tokens_correct = 0
    num_denotation_match = 0
    total_tokens = 0
    for i, ex in enumerate(test_data):
        pred_y_toks = selected_derivs[i].y_toks if i < len(selected_derivs) else [""]
        if i % example_freq == example_freq - 1:
            print('Example %d' % i)
            print('  x      = "%s"' % ex.x)
            print('  y_tok  = "%s"' % ex.y_tok)
            print('  y_pred = "%s"' % pred_y_toks)
        # Compute accuracy metrics
        y_pred = ' '.join(pred_y_toks)
        # Check exact match
        if y_pred == ' '.join(ex.y_tok):
            num_exact_match += 1
        # Check position-by-position token correctness
        num_tokens_correct += sum(a == b for a, b in zip(pred_y_toks, ex.y_tok))
        total_tokens += len(ex.y_tok)
        # Check correctness of the denotation
        if denotation_correct[i]:
            num_denotation_match += 1
    if print_output:
        print("Exact logical form matches: %s" % (render_ratio(num_exact_match, len(test_data))))
        print("Token-level accuracy: %s" % (render_ratio(num_tokens_correct, total_tokens)))
        print("Denotation matches: %s" % (render_ratio(num_denotation_match, len(test_data))))


def render_ratio(numer, denom):
    return "%i / %i = %.3f" % (numer, denom, float(numer) / denom)


def geoquery_preprocess_lf(lf):
    """
    Geoquery preprocessing adapted from Jia and Liang. Standardizes variable names with De Brujin indices -- just a
    smarter way of indexing variables in statements to make parsing easier.
    :param lf:
    :return:
    """
    cur_vars = []
    toks = lf.split(' ')
    new_toks = []
    for w in toks:
        if w.isalpha() and len(w) == 1:
            if w in cur_vars:
                ind_from_end = len(cur_vars) - cur_vars.index(w) - 1
                new_toks.append('V%d' % ind_from_end)
            else:
                cur_vars.append(w)
                new_toks.append('NV')
        else:
            new_toks.append(w)
    return ' '.join(new_toks)
