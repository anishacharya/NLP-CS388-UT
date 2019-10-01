import numpy as np
import common.common_config as common_conf
from typing import Dict, List

from common.utils.indexer import Indexer


class WordEmbedding:
    def __init__(self, pre_trained_embedding_filename: str, word_indexer: Indexer):
        self.embedding_file = pre_trained_embedding_filename
        self.word_ix = word_indexer
        self.ix2embed = self.load_word_embedding()

    def load_word_embedding(self) -> Dict:
        """
        Read a GloVe txt file.we return dictionary
        TODO: Extend for other embeddings
        mapping index to embedding vector( index_to_embedding),
        relativize to train_data i.e. for words in the word indexer
        """
        index_to_embedding = {}

        with open(self.embedding_file, 'r') as glove_file:
            for (i, line) in enumerate(glove_file):
                split = line.split(' ')
                word = split[0]
                ix = self.word_ix.get(word.lower())
                if ix is not None:
                    representation = split[1:]
                    representation = np.array([float(val) for val in representation])
                    index_to_embedding[ix] = list(representation)
        unk = self.word_ix[common_conf.UNK_TOKEN]
        index_to_embedding[unk] = [0.0] * len(representation)  # Empty representation for unknown words.

        return index_to_embedding

    def get_word_embedding(self, word) -> List:
        """
        Given a word returns the corresponding embedding vector
        """
        ix = self.word_ix[word]
        if ix in self.ix2embed:
            word_embed = self.ix2embed[ix]
        else:
            word_embed = self.ix2embed[self.word_ix[common_conf.UNK_TOKEN]]
        return word_embed


class SentenceEmbedding:
    def __init__(self, ix2embedding: Dict, word2ix: Indexer):
        self.ix2embed = ix2embedding
        self.word2ix = word2ix
        self.embed_dim = len(self.ix2embed[0])

    def average_word_embedding(self, sentence: List, word_dropout='False') -> List:
        """
        Implements Deep Averaging Network.
        Deep Unordered Composition Rivals Syntactic Methods
        for Text Classification
        Mohit Iyyer, Varun Manjunatha, Jordan Boyd-Graber, Hal Daume III
        https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf
        """
        embed_vector = [0] * self.embed_dim
        for word in sentence:
            if word in self.word2ix:
                ix = self.word2ix[word]
            else:
                ix = self.word2ix[common_conf.UNK_TOKEN]
            if ix in self.ix2embedding:
                embed_vector = [sum(x) for x in zip(embed_vector, self.ix2embedding[ix])]
            else:
                embed_vector = [sum(x) for x in zip(embed_vector,
                                                    self.ix2embedding[self.word2ix[common_conf.UNK_TOKEN]])]

        sentence_embedding = [i/len(sentence) for i in embed_vector]
        return sentence_embedding

    def get_average_context_embedding(self, tokens: List, idx: int, window_len: int, left=False, right=False):
        """
        Creates a padded seq and gives windowed sentence embedding using DAN

        """
        padded_seq = []
        # pad before
        for i in range(0, window_len):
            padded_seq.append(common_conf.UNK_TOKEN)
        # append tokens
        for token in tokens:
            padded_seq.append(token.lower() if token is not common_conf.UNK_TOKEN else token)
        # pad after
        for i in range(0, window_len):
            padded_seq.append(common_conf.UNK_TOKEN)

        ix_curr = idx + window_len
        ix_lb = ix_curr - window_len
        ix_ub = ix_curr + window_len

        if left is True:
            sentence = padded_seq[ix_lb:ix_curr + 1]
            return self.self.average_word_embedding(sentence=sentence)

        if right is True:
            sentence = padded_seq[ix_curr:ix_ub + 1]
            return self.average_word_embedding(sentence=sentence)

        sentence = padded_seq[ix_lb:ix_ub + 1]
        return self.average_word_embedding(sentence=sentence)

    def skip_thought(self):
        pass

    def sif_arora(self):
        # https://github.com/PrincetonML/SIF_mini_demo
        pass

