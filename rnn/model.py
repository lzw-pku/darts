import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from genotypes import STEPS
from utils import mask2d
from utils import LockedDropout
from utils import embedded_dropout
from torch.autograd import Variable

INITRANGE = 0.04

class LinearLowRank(nn.Module):
    def __init__(self, row, col):
        torch.nn.Module.__init__(self)
        rank = min(row, col)
        self.U = nn.Parameter(torch.Tensor(row, row).uniform_(-INITRANGE, INITRANGE))
        self.sigma = nn.Parameter(torch.Tensor(rank).uniform_(-INITRANGE, INITRANGE))
        if row > col:
            self.pad = torch.zeros(row - col, dtype=self.sigma.dtype, requires_grad=False)
            def dot(input):
                pad_sig = torch.cat([self.sigma, self.pad])
                return input * pad_sig
            self.dot = dot
        elif row < col:
            def dot(input):
                return (input * self.sigma).view(
                    -1, input.size()[-1])[::, :col].view(list(input.size())[:-1]+[col])
            self.dot = dot
        else:
            self.dot = lambda x: x * self.sigma
        self.V = nn.Parameter(torch.Tensor(col, col).uniform_(-INITRANGE, INITRANGE))

    def forward(self, input):
        t1 = input * self.U
        t2 = self.dot(t1)
        t3 = t2 * self.V
        return t3

    def sparse(self):
        return torch.sum(torch.abs(self.sigma))

    def orth(self):
        def compute(u):
            return (u.transpose(1, 0).mm(u) - torch.eye(u.size(0)).cuda()).norm(2)
        return compute(self.U) + compute(self.V)
        #return (self.U.weight.transpose(1, 0).mm(self.U.weight) - torch.eye(self.U.weight.size(0)).cuda()).norm(2)

class DARTSCell(nn.Module):

  def __init__(self, ninp, nhid, dropouth, dropoutx, genotype):
    super(DARTSCell, self).__init__()
    self.nhid = nhid
    self.dropouth = dropouth
    self.dropoutx = dropoutx
    self.genotype = genotype

    # genotype is None when doing arch search
    steps = len(self.genotype.recurrent) if self.genotype is not None else STEPS
    '''
    self._W0 = nn.Parameter(torch.Tensor(ninp+nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE))
    self._Ws = nn.ParameterList([
        nn.Parameter(torch.Tensor(nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE)) for i in range(steps)
    ])
    '''
    self._W0 = LinearLowRank(ninp+nhid, 2*nhid)
    self._Ws = nn.ParameterList([
        LinearLowRank(nhid, 2*nhid) for i in range(steps)
    ])

  def forward(self, inputs, hidden):
    T, B = inputs.size(0), inputs.size(1)

    if self.training:
      x_mask = mask2d(B, inputs.size(2), keep_prob=1.-self.dropoutx)
      h_mask = mask2d(B, hidden.size(2), keep_prob=1.-self.dropouth)
    else:
      x_mask = h_mask = None

    hidden = hidden[0]
    hiddens = []
    for t in range(T):
      hidden = self.cell(inputs[t], hidden, x_mask, h_mask)
      hiddens.append(hidden)
    hiddens = torch.stack(hiddens)
    return hiddens, hiddens[-1].unsqueeze(0)

  def _compute_init_state(self, x, h_prev, x_mask, h_mask):
    if self.training:
      xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
    else:
      xh_prev = torch.cat([x, h_prev], dim=-1)
    c0, h0 = torch.split(self._W0(xh_prev), self.nhid, dim=-1) # !!!!!!!!!!!!!!!!!!!!!
    c0 = c0.sigmoid()
    h0 = h0.tanh()
    s0 = h_prev + c0 * (h0-h_prev)
    return s0

  def _get_activation(self, name):
    if name == 'tanh':
      f = F.tanh
    elif name == 'relu':
      f = F.relu
    elif name == 'sigmoid':
      f = F.sigmoid
    elif name == 'identity':
      f = lambda x: x
    else:
      raise NotImplementedError
    return f

  def cell(self, x, h_prev, x_mask, h_mask):
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)

    states = [s0]
    for i, (name, pred) in enumerate(self.genotype.recurrent):
      s_prev = states[pred]
      if self.training:
        ch = self._Ws[i](s_prev * h_mask)  # !!!!!!!!!!!!!!!!!!
      else:
        ch = self._Ws[i](s_prev) # !!!!!!!!!!!!!!!!!!!!!!!!!
      c, h = torch.split(ch, self.nhid, dim=-1)
      c = c.sigmoid()
      fn = self._get_activation(name)
      h = fn(h)
      s = s_prev + c * (h-s_prev)
      states += [s]
    output = torch.mean(torch.stack([states[i] for i in self.genotype.concat], -1), -1)
    return output


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nhidlast, 
                 dropout=0.5, dropouth=0.5, dropoutx=0.5, dropouti=0.5, dropoute=0.1,
                 cell_cls=DARTSCell, genotype=None):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)
        
        assert ninp == nhid == nhidlast
        if cell_cls == DARTSCell:
            assert genotype is not None
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, genotype)]
        else:
            assert genotype is None
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx)]

        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(ninp, ntoken)
        self.decoder.weight = self.encoder.weight
        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.ntoken = ntoken
        self.cell_cls = cell_cls

    def init_weights(self):
        self.encoder.weight.data.uniform_(-INITRANGE, INITRANGE)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)

    def forward(self, input, hidden, return_h=False):
        batch_size = input.size(1)

        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        logit = self.decoder(output.view(-1, self.ninp))
        log_prob = nn.functional.log_softmax(logit, dim=-1)
        model_output = log_prob
        model_output = model_output.view(-1, batch_size, self.ntoken)

        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden

    def init_hidden(self, bsz):
      weight = next(self.parameters()).data
      return [Variable(weight.new(1, bsz, self.nhid).zero_())]

