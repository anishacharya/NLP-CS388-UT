import torch.nn as nn


class RNN(nn.Module):
    def __init__(self,
                 conf,
                 vocab_size,
                 weights_init=None,
                 emb_dim=300
                 ):
        super(RNN, self).__init__()
        self.conf = conf
        self.vocab_size = vocab_size
        self.weights_init = weights_init
        self.emb_dim = emb_dim
        self.rec_unit = conf.rec_unit
        self.hidden_rnn = conf.hidden_lstm
