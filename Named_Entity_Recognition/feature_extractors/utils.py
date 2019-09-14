from data_utils.nerdata import PersonExample
from utils.utils import Indexer
from typing import List
from nltk.corpus import stopwords
from string import punctuation
import numpy as np


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
    unk = word2index_vocab['__UNK__']
    index_to_embedding[unk] = [0.0] * len(representation)  # Empty representation for unknown words.

    return index_to_embedding


