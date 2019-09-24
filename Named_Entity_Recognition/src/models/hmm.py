from src.data_utils.definitions import Indexer, Token, LabeledSentence, chunks_from_bio_tag_seq
from src.models.utils import ProbabilisticSequenceScorer, dptable

from typing import List


class HmmNerModel(object):
    """
    HMM NER model for predicting tags

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs,
                 transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs
        self.sequence_scorer = ProbabilisticSequenceScorer(tag_indexer=tag_indexer, word_indexer=word_indexer,
                                                           transition_log_probs=transition_log_probs,
                                                           init_log_probs=init_log_probs,
                                                           emission_log_probs=emission_log_probs)

    def decode(self, sentence_tokens: List[Token]):
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        # Implementation of Viterbi Decoding
        # ~\cite{https://en.wikipedia.org/wiki/Viterbi_algorithm}

        # initialize the probability matrix/lattice
        state_space_dim = len(self.tag_indexer)
        time_stamps = len(sentence_tokens)
        viterbi_lattice = [{}]

        # initialize the first step P(t[i]|w0) = p(w0|t[i]) * P(t[i]) = emission[i,index(w0)]
        for state_ix in range(0, state_space_dim):
            prob = self.sequence_scorer.score_init(tag_idx=state_ix) \
                   + self.sequence_scorer.score_emission(sentence_tokens=sentence_tokens, tag_idx=state_ix, word_posn=0)
            viterbi_lattice[0][state_ix] = {"prob": prob, "prev": None}

        for t in range(1, time_stamps):
            viterbi_lattice.append({})
            for state_ix in range(0, state_space_dim):
                max_tr_prob = viterbi_lattice[t-1][0]["prob"] + \
                              self.sequence_scorer.score_transition(prev_tag_idx=0, curr_tag_idx=state_ix)
                prev_st_selected = 0
                for prev_st_ix in range(1, state_space_dim):
                    tr_prob = viterbi_lattice[t-1][prev_st_ix]["prob"] + \
                              self.sequence_scorer.score_transition(prev_tag_idx=prev_st_ix, curr_tag_idx=state_ix)
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st_ix
                max_prob = max_tr_prob + self.sequence_scorer.score_emission(sentence_tokens=sentence_tokens,
                                                                             tag_idx=state_ix, word_posn=t)
                viterbi_lattice[t][state_ix] = {"prob": max_prob, "prev": prev_st_selected}

        # for line in dptable(viterbi_lattice):
        #     print(line)

        most_likely_seq = []
        max_prob = max(value["prob"] for value in viterbi_lattice[-1].values())
        previous = None

        # Get most probable state and its backtrack
        for st, data in viterbi_lattice[-1].items():
            if data["prob"] == max_prob:
                most_likely_seq.append(st)
                previous = st
                break

        # Follow the backtrack till the first observation
        for t in range(len(viterbi_lattice) - 2, -1, -1):
            most_likely_seq.insert(0, viterbi_lattice[t + 1][previous]["prev"])
            previous = viterbi_lattice[t + 1][previous]["prev"]

        pred_tags = []
        for ix in most_likely_seq:
            pred_tags.append(self.tag_indexer.ints_to_objs[ix])
        print('The steps of states are ' + ' '.join(pred_tags) + ' with highest log probability of %s' % max_prob)

        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))
        # raise Exception("IMPLEMENT ME")
