from collections import Counter
from typing import List
from src.data_utils.definitions import PersonExample, Indexer

from nltk.corpus import stopwords
from string import punctuation


def create_index(ner_exs: List[PersonExample]) -> [Indexer, Indexer]:
    stops = set(stopwords.words("english"))
    stops.update(set(punctuation))
    stops.update({'-X-', ',', '$', ':', '-DOCSTART-'})
    # stops = set()
    word_ix = Indexer()
    pos_ix = Indexer()

    # create index for unseen objects
    word_ix.add_and_get_index('__UNK__')
    pos_ix.add_and_get_index('__UNK__')

    for ex in ner_exs:
        for idx in range(0, len(ex)):
            token = ex.tokens[idx].word
            pos = ex.tokens[idx].pos
            if token not in stops:
                word_ix.add_and_get_index(token.lower())
            if pos not in stops:
                pos_ix.add_and_get_index(pos)

    return word_ix, pos_ix


def get_word_index(word_indexer: Indexer, word_counter: Counter, word: str) -> int:
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: Indexer mapping words to indices for HMM featurization
    :param word_counter: Counter containing word counts of training set
    :param word: string word
    :return: int of the word index
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("__UNK__")
    else:
        return word_indexer.add_and_get_index(word)


def inverse_idx_sentence(ix_sentence, ix2word):
    sentence = []
    for ix_token in ix_sentence:
        token = ix2word[ix_token]
        sentence.append(token)
    return sentence


def index_data(ner_exs: List[PersonExample], word_ix, pos_ix):
    # convert data to index
    Sentences = []
    POS = []
    indexed_y = []

    for sent in ner_exs:
        s = []
        pos = []
        indexed_y.append(sent.labels)
        for token in sent.tokens:
            if token.word.lower() in word_ix.objs_to_ints:
                s.append(word_ix.objs_to_ints[token.word.lower()])
            else:
                s.append(word_ix.objs_to_ints['__UNK__'])

            if token.pos in pos_ix.objs_to_ints:
                pos.append(pos_ix.objs_to_ints[token.pos])
            else:
                pos.append(pos_ix.objs_to_ints['__UNK__'])
        Sentences.append(s)
        POS.append(pos)
    return Sentences, POS, indexed_y





