from data_utils.nerdata import PersonExample
from utils.utils import Indexer
from typing import List
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
                word_ix.add_and_get_index(token)
            if pos not in stops:
                pos_ix.add_and_get_index(pos)

    return word_ix, pos_ix