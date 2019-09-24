import torch
import torch.nn as nn

import src.config as conf
from src.data_utils.definitions import LabeledSentence, chunks_from_bio_tag_seq
from src.feature_extractors.lstm_feat_extractor import LSTMFeatureExtractor
from src.models.crf import CRF
from src.models.utils import prepare_data_point


class CrfNerModel(nn.Module):
    def __init__(self, word_ix, tag_ix, embedding_dim, hidden_dim):
        super(CrfNerModel, self).__init__()
        self.word_ix = word_ix
        self.tag_ix = tag_ix
        self.vocab_size = len(word_ix)
        self.label_space = len(tag_ix)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.BOS_TAG_ID = tag_ix.objs_to_ints[conf.BOS_TOKEN]
        self.EOS_TAG_ID = tag_ix.objs_to_ints[conf.EOS_TOKEN]
        self.PAD_TAG_ID = tag_ix.objs_to_ints[conf.PAD_TOKEN]
        self.PAD_ID = word_ix.objs_to_ints[conf.PAD_TOKEN]
        self.lstm_feature_extractor = LSTMFeatureExtractor(vocab_size=self.vocab_size, label_space=self.label_space,
                                                           embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim)
        self.crf = CRF(nb_labels=self.label_space, bos_tag_id=self.BOS_TAG_ID, eos_tag_id=self.EOS_TAG_ID,
                       pad_tag_id=self.PAD_TAG_ID, batch_first=True)

    def forward(self, sentence, mask=None):
        lstm_feat = self.lstm_feature_extractor(sentence)

        return self.crf.decode(lstm_feat, mask=mask)

    def nll(self, x, y, mask=None):
        lstm_feat = self.lstm_feature_extractor(x)
        return self.crf(lstm_feat, y, mask=mask)

    def decode(self, sentence_tokens):
        pred_tags = []
        one_hot = [self.word_ix.objs_to_ints[token.word] if token.word in self.word_ix.objs_to_ints
                   else self.word_ix.objs_to_ints[conf.UNK_TOKEN] for token in sentence_tokens]
        x1 = torch.tensor(one_hot, dtype=torch.long)
        x_test = torch.full((1, len(x1)), 0, dtype=torch.long)
        x_test[0, :x1.shape[0]] = x1
        score, seq = self.forward(x_test)
        for tag in seq[0]:
            pred_tags.append(self.tag_ix.ints_to_objs[tag])
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))



