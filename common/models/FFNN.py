import torch.nn as nn


class FFNN(nn.Module):
    def __init__(self, inp, hid, out):
        super(FFNN, self).__init__()
        self.V = nn.Linear(inp, hid)
        self.g = nn.Tanh()
        self.W = nn.Linear(hid, out)
        self.softmax = nn.Softmax(dim=0)
        nn.init.xavier_uniform(self.V.weight)
        nn.init.xavier_uniform(self.W.weight)

    def forward(self, x):
        return self.softmax(self.W(self.g(self.V(x))))
