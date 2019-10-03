import torch.nn as nn


# class FFNN(nn.Module):
#     def __init__(self, inp, hid, out):
#         super(FFNN, self).__init__()
#         self.V = nn.Linear(inp, hid)
#         self.g = nn.Tanh()
#         self.W = nn.Linear(hid, out)
#         self.softmax = nn.Softmax(dim=0)
#         nn.init.xavier_uniform_(self.V.weight)
#         nn.init.xavier_uniform_(self.W.weight)
#
#     def forward(self, x):
#         return self.softmax(self.W(self.g(self.V(x))))

class FFNN(nn.Module):
    def __init__(self, conf):
        super(FFNN, self).__init__()
        self.conf = conf

        self.h1 = nn.Linear(self.conf.input_dim, self.conf.hidden_1)
        self.h2 = nn.Linear(self.conf.hidden_1, self.conf.hidden_2)
        # self.h3 = nn.Linear(self.conf.hidden_2, self.conf.hidden_3)

        self.hidden2tag = nn.Linear(self.conf.hidden_2, self.conf.no_classes)
        self.softmax = nn.Softmax()

        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.conf.dropout)

    def forward(self, x):
        h1 = self.drop(self.act(self.h1(x)))
        h2 = self.drop(self.act(self.h2(h1)))
        # h3 = self.drop(self.act(self.h3(h2)))
        z = self.hidden2tag(h2)

        return self.softmax(z)
