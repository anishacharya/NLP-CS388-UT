import numpy as np
import torch
from torch import nn

import src.config as conf
from src.data_utils.definitions import LabeledSentence, chunks_from_bio_tag_seq
from src.feature_extractors.emission_features import extract_emission_features
from src.utils.utils import flatten
"""
Author: Anish Acharya
Adopted from: 
~/cite https://pytorch.org/tutorials/beginner/nlp
~/cite https://www.cs.utexas.edu/~gdurrett/courses/fa2019/cs388.shtml
"""

class EmissionCrfNerModel(nn.Module):
    def __init__(self, word_ix, tag_ix, feature_cache, feature_dim, feature_ix):
        super(EmissionCrfNerModel, self).__init__()
        self.word_ix = word_ix
        self.tag_ix = tag_ix
        self.feature_cache = feature_cache
        self.nb_features = feature_dim
        self.vocab_size = len(word_ix)
        self.nb_labels = len(tag_ix)
        self.feature_ix = feature_ix

        self.BOS_TAG_ID = tag_ix.objs_to_ints[conf.BOS_TOKEN]
        self.EOS_TAG_ID = tag_ix.objs_to_ints[conf.EOS_TOKEN]
        self.PAD_TAG_ID = tag_ix.objs_to_ints[conf.PAD_TOKEN]
        self.PAD_ID = word_ix.objs_to_ints[conf.PAD_TOKEN]

        self.emission_weights = nn.Parameter(torch.empty(self.nb_features, 1))
        self.transitions = nn.Parameter(torch.empty(self.nb_labels, self.nb_labels))
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.emission_weights, -0.1, 0.1)
        self.transitions.data[:, self.BOS_TAG_ID] = -10000.0
        self.transitions.data[self.EOS_TAG_ID, :] = -10000.0

        if self.PAD_TAG_ID is not None:
            self.transitions.data[self.PAD_TAG_ID, :] = -10000.0
            self.transitions.data[:, self.PAD_TAG_ID] = -10000.0
            self.transitions.data[self.PAD_TAG_ID, self.EOS_TAG_ID] = 0.0
            self.transitions.data[self.PAD_TAG_ID, self.PAD_TAG_ID] = 0.0

    def forward(self, x):
        emission_feat = self.get_emissions(x)
        return self.seq_decode(emission_feat)

    def nll(self, x, y):
        emission_feat = self.get_emissions(x)
        nll = -self.log_likelihood(emission_feat, y)
        return nll

    def get_emissions(self, seq_x):
        emissions = []
        for i, x in enumerate(seq_x):
            ind2 = x.flatten()
            ind1 = np.array([i // 14 for i in range(len(ind2))])
            indices = torch.from_numpy(np.stack((ind1, ind2)))
            values = torch.ones(len(ind2))
            features = torch.sparse.FloatTensor(indices=indices, values=values,
                                                size=torch.Size([self.nb_labels, self.nb_features]))
            potential = torch.sparse.mm(features, self.emission_weights)
            emissions.append(potential.squeeze())
        emissions = torch.stack(emissions).unsqueeze(0)
        return emissions

    def log_likelihood(self, emissions, tags):
        scores = self.compute_scores(emissions, tags)
        partition = self.compute_log_partition(emissions)
        return torch.sum(scores - partition)

    def compute_scores(self, emissions, tags):
        batch_size, seq_length = tags.shape
        scores = torch.zeros(batch_size)
        first_tags = tags[:, 0]

        emm = torch.ones(emissions.shape[:2])
        last_valid_idx = emm.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()
        t_scores = self.transitions[self.BOS_TAG_ID, first_tags]
        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()
        scores += e_scores + t_scores

        for i in range(1, seq_length):
            previous_tags = tags[:, i - 1]
            current_tags = tags[:, i]
            e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
            t_scores = self.transitions[previous_tags, current_tags]
            scores += e_scores + t_scores

        scores += self.transitions[last_tags, self.EOS_TAG_ID]
        return scores

    def compute_log_partition(self, emissions):
        batch_size, seq_length, nb_labels = emissions.shape
        alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]

        for i in range(1, seq_length):
            alpha_t = []

            for tag in range(nb_labels):
                e_scores = emissions[:, i, tag]
                e_scores = e_scores.unsqueeze(1)
                t_scores = self.transitions[:, tag]
                t_scores = t_scores.unsqueeze(0)

                # combine current scores with previous alphas
                scores = e_scores + t_scores + alphas

                # add the new alphas for the current tag
                alpha_t.append(torch.logsumexp(scores, dim=1))

            new_alphas = torch.stack(alpha_t).t()
            alphas = new_alphas

        # add the scores for the final transition
        last_transition = self.transitions[:, self.EOS_TAG_ID]
        end_scores = alphas + last_transition.unsqueeze(0)

        # return a log of sums of exps
        return torch.logsumexp(end_scores, dim=1)

    def seq_decode(self, emissions, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)
        scores, sequences = self._viterbi_decode(emissions, mask)
        return scores, sequences

    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_length, nb_labels = emissions.shape
        # in the first iteration, BOS will have all the scores and then, the max
        alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]
        backpointers = []
        for i in range(1, seq_length):
            e_scores = emissions[:, i].unsqueeze(1)
            t_scores = self.transitions.unsqueeze(0)
            a_scores = alphas.unsqueeze(2)
            scores = e_scores + t_scores + a_scores
            max_scores, max_score_tags = torch.max(scores, dim=1)
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * max_scores + (1 - is_valid) * alphas
            backpointers.append(max_score_tags.t())
        last_transition = self.transitions[:, self.EOS_TAG_ID]
        end_scores = alphas + last_transition.unsqueeze(0)
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)
        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        for i in range(batch_size):
            sample_length = emission_lengths[i].item()
            sample_final_tag = max_final_tags[i].item()
            sample_backpointers = backpointers[: sample_length - 1]
            sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)
            best_sequences.append(sample_path)

        return max_final_scores, best_sequences

    @staticmethod
    def _find_best_path(sample_id, best_tag, back_pointers):
        best_path = [best_tag]
        for back_pointers_t in reversed(back_pointers):
            best_tag = back_pointers_t[best_tag][sample_id].item()
            best_path.insert(0, best_tag)
        return best_path

    def decode(self, sentence_tokens):
        tag_indexer = self.tag_ix
        feature_indexer = self.feature_ix
        all_features = []
        for word_idx in range(0, len(sentence_tokens)):
            features = []
            for tag_idx in range(0, len(tag_indexer)):
                features.append(extract_emission_features(sentence_tokens,
                                                          word_idx, tag_indexer.get_object(tag_idx), feature_indexer,
                                                          add_to_indexer=False))
            all_features.append(features)
        all_features = np.array(all_features)
        score, seq = self.forward(all_features)
        seq = flatten(seq)
        pred_tags = []
        for j in seq:
            pred_tags.append(self.tag_ix.ints_to_objs[j])
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))