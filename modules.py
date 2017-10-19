
import torch as T
import torch.nn as NN
import torch.nn.functional as F
import torch.nn.init as INIT
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

def tovar(*arrs):
    tensors = [(T.Tensor(a.astype('float32')) if isinstance(a, np.ndarray) else a) for a in arrs]
    vars_ = [T.autograd.Variable(t) for t in tensors]
    return vars_[0] if len(vars_) == 1 else vars_


def tonumpy(*vars_):
    arrs = [(v.data.cpu().numpy() if isinstance(v, T.autograd.Variable) else
             v.cpu().numpy() if T.is_tensor(v) else v) for v in vars_]
    return arrs[0] if len(arrs) == 1 else arrs


def div_roundup(x, d):
    return (x + d - 1) / d
def roundup(x, d):
    return (x + d - 1) / d * d


def log_sigmoid(x):
    return -F.softplus(-x)
def log_one_minus_sigmoid(x):
    return -x - F.softplus(-x)

def binary_cross_entropy_with_logits_per_sample(input, target, weight=None):
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    return loss.sum(1)


def advanced_index(t, dim, index):
    return t.transpose(dim, 0)[index].transpose(dim, 0)


def length_mask(size, length):
    length = tonumpy(length)
    batch_size = size[0]
    weight = T.zeros(*size)
    for i in range(batch_size):
        weight[i, :length[i]] = 1.
    weight = tovar(weight)
    return weight


def dynamic_rnn(rnn, seq, length, initial_state):
    length[length==0] = 1
    length_sorted, length_sorted_idx = T.sort(length, 0, descending=True)
    _, length_inverse_idx = T.sort(length_sorted_idx)
    rnn_in = pack_padded_sequence(
            advanced_index(seq, 1, length_sorted_idx),
            tonumpy(length_sorted),
            )
    rnn_out, rnn_last_state = rnn(rnn_in, initial_state)
    rnn_out = pad_packed_sequence(rnn_out)[0]
    out = advanced_index(rnn_out, 1, length_inverse_idx)
    if isinstance(rnn_last_state, tuple):
        state = tuple(advanced_index(s, 1, length_inverse_idx) for s in rnn_last_state)
    else:
        state = advanced_index(s, 1, length_inverse_idx)

    return out, state


def check_grad(params):
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.data
        anynan = (g != g).long().sum()
        anybig = (g.abs() > 1e+5).long().sum()
        if anynan or anybig:
            return False
    return True


def clip_grad(params, clip_norm):
    norm = np.sqrt(
            sum(p.grad.data.norm() ** 2
                for p in params if p.grad is not None
                )
            )
    if norm > clip_norm:
        for p in params:
            if p.grad is not None:
                p.grad /= norm / clip_norm
    return norm

def init_lstm(lstm):
    for name, param in lstm.named_parameters():
        if name.startswith('weight_ih'):
            INIT.xavier_uniform(param.data)
        elif name.startswith('weight_hh'):
            INIT.orthogonal(param.data)
        elif name.startswith('bias'):
            INIT.constant(param.data, 0)

def init_weights(module):
    for name, param in module.named_parameters():
        if name.find('weight') != -1:
            INIT.xavier_uniform(param.data)
        elif name.find('bias') != -1:
            INIT.constant(param.data, 0)

class Residual(NN.Module):
    def __init__(self,size, relu = True):
        NN.Module.__init__(self)
        self.size = size
        self.linear = NN.Linear(size, size)
        if relu:
            self.relu = NN.LeakyReLU()
        else:
            self.relu = False

    def forward(self, x):
        if self.relu:
            return self.relu(self.linear(x) + x)
        else:
            return self.linear(x) + x

class ConvMask(NN.Module):
    def __init__(self):
        NN.Module.__init__(self)

    def forward(self, x):
        global convlengths
        mask = length_mask((x.size()[0], x.size()[2]),convlengths).unsqueeze(1)
        x = x * mask
        return x
