import torch
import torch.nn as nn


class LSTMFeatureExtractor(nn.Module):
    """
    This class defines a simple LSTM architecture to extract
    ~/cite{https://pytorch.org/tutorials/beginner}
    ~/cite{http://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf}

    """
    def __init__(self, vocab_size, label_space, embedding_dim, hidden_dim):
        super(LSTMFeatureExtractor, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.label_space = label_space
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.label_space)
        # self.hidden = self.init_hidden()
        self.hidden = None

    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2),
                torch.randn(2, batch_size, self.hidden_dim // 2))

    def forward(self, sentence_batch):
        self.hidden = self.init_hidden(sentence_batch.shape[0])
        embeds = self.word_embeds(sentence_batch)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats


