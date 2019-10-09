import common.common_config as common_conf
from typing import Dict, List
import time
import numpy as np

from common.utils.indexer import Indexer
from common.utils.utils import word_dropout


class WordEmbedding:
    def __init__(self, pre_trained_embedding_filename: str, word_indexer: Indexer):
        self.embedding_file = pre_trained_embedding_filename
        self.word_ix = word_indexer
        self.emb_dim = 0
        self.ix2embed = self.load_and_index_word_embedding()

    def load_and_index_word_embedding(self) -> Dict:
        """
        Read a GloVe txt file.we return dictionary
        TODO: Extend for other embeddings [FastText]
        mapping index to embedding vector( index_to_embedding),
        relativize to train_data i.e. for words in the word indexer
        """
        print(" Loading Glove and Relativizing to Train Data ")
        t = time.time()
        index_to_embedding = {}

        glove_file = open(self.embedding_file, 'r')
        for line in glove_file:
            split = line.split(' ')
            word = split[0]
            representation = split[1:]
            representation = np.array([float(val) for val in representation])
            if self.word_ix.contains(word):
                ix = self.word_ix.add_and_get_index(word)
                index_to_embedding[ix] = representation
        glove_file.close()

        # create empty representation for unknown words.
        self.emb_dim = len(representation)
        _UNK_ix = self.word_ix.add_and_get_index(common_conf.UNK_TOKEN)
        _PAD_ix = self.word_ix.add_and_get_index(common_conf.PAD_TOKEN)

        index_to_embedding[_UNK_ix] = [0.0] * self.emb_dim
        index_to_embedding[_PAD_ix] = [0.0] * self.emb_dim

        for word_ix in self.word_ix.ints_to_objs.keys():
            if word_ix not in index_to_embedding:
                print('No Glove Representation for: {}: Setting to _UNK_ '.format(self.word_ix.ints_to_objs[word_ix]))
                index_to_embedding[word_ix] = index_to_embedding[_UNK_ix]
        print("Time Taken for embedding dict creation: {}".format(time.time()-t))
        return index_to_embedding

    def get_word_embedding(self, word) -> List:
        """
        Given a word returns the corresponding embedding vector
        """
        word = word.lower()
        ix = self.word_ix.add_and_get_index(word) if self.word_ix.contains(word) \
            else self.word_ix.add_and_get_index(common_conf.UNK_TOKEN)

        if ix in self.ix2embed:
            word_embed = self.ix2embed[ix]
        else:
            word_embed = self.ix2embed[self.word_ix.add_and_get_index(common_conf.UNK_TOKEN)]
            print('word |{}| not in glove => returning UNK token'.format(word))
        return word_embed


class SentenceEmbedding:
    def __init__(self, word_embed: WordEmbedding):
        self.ix2embed = word_embed.ix2embed
        self.word2ix = word_embed.word_ix
        self.embed_dim = word_embed.emb_dim

    def average_word_embedding(self, sentence: List[int], word_dropout_rate=0) -> List:

        """
        Implements Deep Averaging Network.
        Deep Unordered Composition Rivals Syntactic Methods
        for Text Classification
        Mohit Iyyer, Varun Manjunatha, Jordan Boyd-Graber, Hal Daume III
        https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf

        :param sentence: indexed sentence i.e. [1,4,7,8]
        where indexes are corresponding indexes of the tokens in the indexer
        :param word_drop: if True then implements random dropout as described in DAN paper
        :param word_dropout_rate: prob with which we drop a word
        :return: sentence embedding vector taken as the average of the words as described in the DAN paper
        """

        embedding_accumulator = [0] * self.embed_dim
        word_count = len(sentence)

        for ix in sentence:
            if word_dropout(word_dropout_rate) and word_count > 3:
                # we don't want to drop words if the sentence is too short
                word_count -= 1
                continue    # skip this word
            if ix not in self.ix2embed:
                # ix = self.word2ix.objs_to_ints[common_conf.UNK_TOKEN]
                # print('word embedding not found for |{}|'.format(self.word2ix.ints_to_objs[ix]))
                raise SystemExit("Should Not be here - Debug Embedding Loader Code")
            embedding_accumulator = [sum(x) for x in zip(embedding_accumulator, self.ix2embed[ix])]

        # if word_count == 0:
        #     sent_string = ''
        #     for ix in sentence: sent_string += self.word2ix.ints_to_objs[ix] + ' '
        #     print('No word Embedding for any word in the sentence |{}| - please check'.format(sent_string))
        #     return self.ix2embed[self.word2ix.objs_to_ints[common_conf.UNK_TOKEN]]

        sentence_embedding = [i / word_count for i in embedding_accumulator]
        return sentence_embedding

    def get_average_context_embedding(self, tokens: List, idx: int, window_len: int, left=False, right=False):
        """
        Creates a padded seq and gives windowed sentence embedding using DAN
        similar to convolution
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
            return self.average_word_embedding(sentence=sentence)

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
