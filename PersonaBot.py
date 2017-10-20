import argparse

from torch.nn import Parameter
from functools import wraps

import datetime

import torch as T
import torch.nn as NN
import torch.nn.functional as F
import torch.nn.init as INIT
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import weight_norm as torch_weight_norm
import numpy as NP
import numpy.random as RNG
import tensorflow as TF     # for Tensorboard
import os
import re
import h5py
import numpy as np

def_size = 64

parser = argparse.ArgumentParser(description='Ubuntu Dialogue dataset parser')
parser.add_argument('--dataroot', type=str,required=True, help='Root of the data downloaded from github')
parser.add_argument('--outputdir', type=str, default ='outputs',required=True, help='output directory')
parser.add_argument('--logdir', type=str, default='logs', help='log directory')
parser.add_argument('--encoder_layers', type=int, default=2)
parser.add_argument('--decoder_layers', type=int, default=2)
parser.add_argument('--context_size', type=int, default=def_size)
parser.add_argument('--encoder_state_size', type=int, default=def_size)
parser.add_argument('--decoder_state_size', type=int, default=def_size)
parser.add_argument('--user_size', type=int, default=45)
parser.add_argument('--word_size', type=int, default=123)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--gradclip', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--modelname', type=str, default = '')
parser.add_argument('--modelnamesave', type=str, default='')
parser.add_argument('--modelnameload', type=str, default='')
parser.add_argument('--loaditerations', type=int, default=0)
args = parser.parse_args()
import pickle

train_file = h5py.File(args.outputdir + 'dataset_train.h5', 'r')
val_file = h5py.File(args.outputdir + 'dataset_val.h5', 'r')
test_file = h5py.File(args.outputdir + 'dataset_test.h5', 'r')
word2idx, idx2word = pickle.load(open(args.outputdir +  "word_dicts.p", "r" ))
user2idx, idx2user = pickle.load(open(args.outputdir +  "user_dicts.p", "r" ))
try:
    os.mkdir(args.logdir)
except:
    pass

if len(args.modelname) > 0:
    modelnamesave = args.modelname
    modelnameload = None
else:
    modelnamesave = args.modelnamesave
    modelnameload = args.modelnameload


def logdirs(logdir, modelnamesave):
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    logdir = (
            logdir + '/%s-%s' % 
            (modelnamesave, datetime.datetime.strftime(
                datetime.datetime.now(), '%Y%m%d%H%M%S')
                )
            )
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    elif not os.path.isdir(logdir):
        raise IOError('%s is not a directory' % logdir)
    return logdir
log_train_d = logdirs(args.logdir, args.modelnamesave)




def tovar(*arrs):
    tensors = [(T.Tensor(a.astype('float32')) if isinstance(a, NP.ndarray) else a).cuda() for a in arrs]
    vars_ = [T.autograd.Variable(t) for t in tensors]
    return vars_[0] if len(vars_) == 1 else vars_


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


def tonumpy(*vars_):
    arrs = [(v.data.cpu().numpy() if isinstance(v, T.autograd.Variable) else
             v.cpu().numpy() if T.is_tensor(v) else v) for v in vars_]
    return arrs[0] if len(arrs) == 1 else arrs


def advanced_index(t, dim, index):
    return t.transpose(dim, 0)[index].transpose(dim, 0)

def dynamic_rnn(rnn, seq, length, initial_state):
    length_sorted, length_sorted_idx = T.sort(length, descending=True)
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

uni_words = word2idx.keys()
num_words = len(uni_words)
uni_users = user2idx.keys()
num_users = len(uni_users)


wordencoder = NN.Embedding(num_words, args.word_size)
userencoder = NN.Embedding(num_users, args.user_size)



class Encoder(NN.Module):
    def __init__(self,
                 state_size=def_size,
                 word_size=def_size,
                 context_size=def_size,
                 user_size = def_size,
                 num_layers=2):
        NN.Module.__init__(self)
        self._state_size = state_size
        self._word_size = word_size
        self._num_layers = num_layers
        self._context_size = context_size
        self._user_size = user_size
        
        self.rnn = NN.LSTM(
                input_size=word_size + context_size + user_size,
                hidden_size = state_size//2,
                num_layers=num_layers,
                bidirectional=True,
                )
        init_lstm(self.rnn)
        

    def forward(self, sentences, users, wordencoder=wordencoder, userencoder=userencoder):
        state_size = self._state_size
        num_layers = self._num_layers
        batch_size, max_sentlen = sentences.size()
        assert users.size() == batch_size
        
        initial_state = (
                tovar(T.zeros(num_layers * 2, batch_size, state_size // 2)),
                tovar(T.zeros(num_layers * 2, batch_size, state_size // 2)),
                )
        word_emb = wordencoder(sentences)
        user_emb = userencoder(users)
        return



itr = args.loaditerations
batchsize = args.batchsize
trn_conversations = train_file.keys()
batch_idxes = NP.random.choice(trn_conversations,batchsize)
for idx in batch_idxes:
    conversation = train_file[idx].value
    users = conversation[:,0]
    lengths = conversation[:,-1]
    messages = conversation[:,1:-1]
e = Encoder()

e(x, userencoder, wordencoder)
d = Decoder()


while True:
    a = 2



a = 2


