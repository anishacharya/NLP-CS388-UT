import torch
from torch import nn
import numpy as np

"""
Author: Anish Acharya
Adopted from: 
~/cite https://pytorch.org/tutorials/beginner/nlp
~/cite https://www.cs.utexas.edu/~gdurrett/courses/fa2019/cs388.shtml
"""


class CRF(nn.Module):
    def __init__(
        self, nb_labels, bos_tag_id, eos_tag_id, nb_features, pad_tag_id=None, batch_first=True,
    ):
        super().__init__()

        self.nb_labels = nb_labels
        self.BOS_TAG_ID = bos_tag_id
        self.EOS_TAG_ID = eos_tag_id
        self.PAD_TAG_ID = pad_tag_id
        self.batch_first = batch_first
        self.nb_features = nb_features

        self.transitions = nn.Parameter(torch.empty(self.nb_labels, self.nb_labels))

        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        self.transitions.data[:, self.BOS_TAG_ID] = -10000.0
        self.transitions.data[self.EOS_TAG_ID, :] = -10000.0

        if self.PAD_TAG_ID is not None:
            self.transitions.data[self.PAD_TAG_ID, :] = -10000.0
            self.transitions.data[:, self.PAD_TAG_ID] = -10000.0
            self.transitions.data[self.PAD_TAG_ID, self.EOS_TAG_ID] = 0.0
            self.transitions.data[self.PAD_TAG_ID, self.PAD_TAG_ID] = 0.0

    def forward(self, emissions, tags, mask=None):
        nll = -self.log_likelihood(emissions, tags, mask=mask)
        return nll

    def log_likelihood(self, emissions, tags, mask=None):
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)

        scores = self._compute_scores(emissions, tags, mask=mask)
        partition = self._compute_log_partition(emissions, mask=mask)
        return torch.sum(scores - partition)

    def decode(self, emissions, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)

        scores, sequences = self._viterbi_decode(emissions, mask)
        return scores, sequences

    def _compute_scores(self, emissions, tags, mask):
        batch_size, seq_length = tags.shape
        scores = torch.zeros(batch_size)
        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()
        t_scores = self.transitions[self.BOS_TAG_ID, first_tags]
        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()
        scores += e_scores + t_scores
        for i in range(1, seq_length):
            is_valid = mask[:, i]
            previous_tags = tags[:, i - 1]
            current_tags = tags[:, i]
            e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
            t_scores = self.transitions[previous_tags, current_tags]
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid
            scores += e_scores + t_scores
        scores += self.transitions[last_tags, self.EOS_TAG_ID]
        return scores

    def _compute_log_partition(self, emissions, mask):
        batch_size, seq_length, nb_labels = emissions.shape
        alphas = self.transitions[self.BOS_TAG_ID, :].unsqueeze(0) + emissions[:, 0]
        for i in range(1, seq_length):
            e_scores = emissions[:, i].unsqueeze(1)
            t_scores = self.transitions.unsqueeze(0)
            a_scores = alphas.unsqueeze(2)
            scores = e_scores + t_scores + a_scores
            new_alphas = torch.logsumexp(scores, dim=1)
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas
        last_transition = self.transitions[:, self.EOS_TAG_ID]
        end_scores = alphas + last_transition.unsqueeze(0)

        return torch.logsumexp(end_scores, dim=1)

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

    def _find_best_path(self, sample_id, best_tag, backpointers):
        best_path = [best_tag]
        for backpointers_t in reversed(backpointers):
            best_tag = backpointers_t[best_tag][sample_id].item()
            best_path.insert(0, best_tag)
        return best_path
