
import torch as T
import torch.nn as NN
import torch.nn.functional as F
import torch.nn.init as INIT
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os

import numpy as np

def cuda(obj):
    if os.getenv('USE_CUDA', None):
        return obj.cuda()
    else:
        return obj

def tovar(*arrs):
    tensors = [(T.Tensor(a.astype('float32')) if isinstance(a, np.ndarray) else a) for a in arrs]
    vars_ = [T.autograd.Variable(t) for t in tensors]
    if os.getenv('USE_CUDA', None):
        vars_ = [v.cuda() for v in vars_]
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


class HierarchicalLogSoftmax(NN.Module):
    def __init__(self, input_size, n_classes, n_words):
        '''
        n_words must be dividable by n_classes
        '''
        NN.Module.__init__(self)
        assert n_words % n_classes == 0
        self.n_words_per_cls = n_words // n_classes
        self.n_classes = n_classes

        self.W_cls = NN.Parameter(T.zeros(input_size, n_classes))
        self.b_cls = NN.Parameter(T.zeros(n_classes))
        self.W_word_in_cls = NN.Parameter(T.zeros(n_classes, input_size, self.n_words_per_cls))
        self.b_word_in_cls = NN.Parameter(T.zeros(n_classes, self.n_words_per_cls))

        INIT.xavier_uniform(self.W_cls)
        INIT.xavier_uniform(self.W_word_in_cls)
        INIT.constant(self.b_cls, 0)
        INIT.constant(self.b_word_in_cls, 0)

        # self.mapping[x]
        perm = np.random.permutation(n_words)
        mapping_dict = dict(zip(range(n_words), perm))
        inv_mapping_dict = dict(zip(perm, range(n_words)))
        self.mapping = T.LongTensor(np.array([mapping_dict[i] for i in range(n_words)], dtype='int64'))
        self.inv_mapping = T.LongTensor(np.array([inv_mapping_dict[i] for i in range(n_words)], dtype='int64'))

    def forward(self, x, target=None):
        '''
        x: (batch_size, input_size)
        target: LongTensor (batch_size,) or None

        return:
        if target is None, returns
            prob: (batch_size, n_words)
        if target is a LongTensor, returns
            prob: (batch_size)
        '''
        batch_size = x.size()[0]
        cls_prob = F.log_softmax(x @ self.W_cls + self.b_cls)
        if target is None:
            word_in_cls_logit = x.unsqueeze(0) @ self.W_word_in_cls + self.b_word_in_cls.unsqueeze(1)
            word_in_cls_prob = F.log_softmax(word_in_cls_logit.view(self.n_classes * batch_size, -1))
            word_in_cls_prob = word_in_cls_prob.view(self.n_classes, batch_size, -1)
            word_prob = cls_prob.transpose(1, 0).unsqueeze(2) + word_in_cls_prob
            word_prob = word_prob.transpose(1, 0).contiguous().view(batch_size, -1)

            return word_prob[:, self.mapping]
        else:
            internal_target = tovar(self.mapping)[target]
            internal_target_cls = internal_target / self.n_words_per_cls
            internal_target_word_in_cls = internal_target % self.n_words_per_cls

            target_cls_prob = cls_prob.gather(1, internal_target_cls.unsqueeze(1))
            target_W = self.W_word_in_cls[internal_target_cls]
            target_b = self.b_word_in_cls[internal_target_cls]
            target_word_in_cls_prob = F.log_softmax((x.unsqueeze(1) @ target_W).squeeze(1) + target_b)

            word_prob = target_cls_prob + target_word_in_cls_prob
            word_prob = word_prob.gather(1, internal_target_word_in_cls.unsqueeze(1))

            return word_prob