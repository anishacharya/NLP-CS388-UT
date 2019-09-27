from src.data_utils.definitions import Indexer, Token, LabeledSentence, chunks_from_bio_tag_seq
from src.data_utils.utils import inverse_idx_sentence
from src.feature_extractors.embedding_features import get_context_vector
from src.feature_extractors.indicator_features import pos_indicator_feat,is_upper_indicator_feat,all_caps_indicator_feat
import torch
from typing import List, Dict


class MLPNerClassifier(object):
    """
    Classifier to classify a token in a sentence as a PERSON token or not.
    Constructor arguments are merely suggestions; you're free to change these.
    """

    def __init__(self,
                 model,
                 word_ix: Indexer,
                 pos_ix: Indexer,
                 tag_ix: Indexer,
                 ix2embed: Dict):
        self.model = model
        self.word_ix = word_ix
        self.pos_ix = pos_ix
        self.tag_ix = tag_ix
        self.ix2embed = ix2embed

    def decode(self, tokens: List[Token]):
        """
        Makes a prediction for token at position idx in the given PersonExample
        Returns 0 if not a person token, 1 if a person token
        """
        pred_out = []

        for idx, token in enumerate(tokens):
            "Word Indicator"
            # word_indicator_feat_dim = len(self.word_ix)
            # word_indicator = [0] * word_indicator_feat_dim
            # if token.word.lower() in self.word_ix.objs_to_ints:
            #     ix = self.word_ix.objs_to_ints[token.word.lower()]
            #     word_indicator[ix] = 1
            # else:
            #     word_indicator[self.word_ix.objs_to_ints['__UNK__']] = 1

            "pos indicator"
            pos_indicator_feat_dim = len(self.pos_ix)
            pos_indicator = [0] * pos_indicator_feat_dim
            if token.pos in self.pos_ix.objs_to_ints:
                ix = self.pos_ix.objs_to_ints[token.pos]
                pos_indicator[ix] = 1
            else:
                pos_indicator[self.pos_ix.objs_to_ints['__UNK__']] = 1

            "starts with a capital"
            is_upper = is_upper_indicator_feat(word=token.word, idx=idx)
            # # all caps
            is_all_caps = all_caps_indicator_feat(word=token.word)

            " Current word Embedding "
            if token.word.lower() in self.word_ix.objs_to_ints:
                token_ix = self.word_ix.objs_to_ints[token.word.lower()]
            else:
                token_ix = self.word_ix.objs_to_ints['__UNK__']
            if token_ix in self.ix2embed:
                word_emb = self.ix2embed[token_ix]
            else:
                word_emb = self.ix2embed[self.word_ix.objs_to_ints['__UNK__']]

            # Context vector
            sentence = [token.word for token in tokens]
            context_window_1 = get_context_vector(tokens=sentence,
                                                  idx=idx,
                                                  window_len=1,
                                                  word2ix=self.word_ix.objs_to_ints,
                                                  ix2embed=self.ix2embed)

            # context_window_2 = get_context_vector(tokens=sentence,
            #                                       idx=idx,
            #                                       window_len=2,
            #                                       word2ix=self.word_ix.objs_to_ints,
            #                                       ix2embed=self.ix2embed)
            # context_left_1 = get_context_vector(tokens=sentence,
            #                                     idx=idx,
            #                                     window_len=1,
            #                                     word2ix=self.word_ix.objs_to_ints,
            #                                     ix2embed=self.ix2embed,
            #                                     left=True)
            # context_left_2 = get_context_vector(tokens=sentence,
            #                                     idx=idx,
            #                                     window_len=2,
            #                                     word2ix=self.word_ix.objs_to_ints,
            #                                     ix2embed=self.ix2embed,
            #                                     left=True)
            # context_right_1 = get_context_vector(tokens=sentence,
            #                                     idx=idx,
            #                                     window_len=1,
            #                                     word2ix=self.word_ix.objs_to_ints,
            #                                     ix2embed=self.ix2embed,
            #                                     right=True)
            feat_vec = []
            # feat_vec = feat_vec + word_indicator
            feat_vec = feat_vec + pos_indicator
            feat_vec = feat_vec + is_upper
            feat_vec = feat_vec + is_all_caps
            #
            feat_vec = feat_vec + word_emb
            feat_vec = feat_vec + context_window_1
            # feat_vec = feat_vec + context_window_2
            # feat_vec = feat_vec + context_left_1
            # feat_vec = feat_vec + context_left_2
            # feat_vec = feat_vec + context_right_1

            x_test = torch.FloatTensor(feat_vec)
            y_hat_test = self.model(x_test)

            max_ix = -1
            curr_max_score = -1000000
            for ix, score in enumerate(y_hat_test):
                if score > curr_max_score:
                    max_ix = ix
                    curr_max_score = score
            pred_out.append(self.tag_ix.ints_to_objs[max_ix])

        return LabeledSentence(tokens, chunks_from_bio_tag_seq(pred_out))