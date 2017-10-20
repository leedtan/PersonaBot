
from torch.nn import Parameter
from functools import wraps

import torch as T
import torch.nn as NN
import torch.nn.functional as F
import torch.nn.init as INIT
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy.random as RNG
import tensorflow as TF     # for Tensorboard
from numpy import rate


import argparse, sys, datetime, pickle, os

import matplotlib
from mailbox import _create_carefully
import matplotlib.pyplot as PL

from PIL import Image


from torch.utils.data import DataLoader, Dataset
import numpy as np
np.set_printoptions(suppress=True)
from collections import Counter
from data_loader_stage1 import *

from modules import *


class Encoder(NN.Module):
    def __init__(self,size_usr, size_wd, output_size, num_layers):
        NN.Module.__init__(self)
        self._output_size = output_size
        self._size_wd = size_wd
        self._size_usr = size_usr
        self._num_layers = num_layers

        self.rnn = NN.LSTM(
                size_usr + size_wd,
                output_size // 2,
                num_layers,
                bidirectional=True,
                )
        init_lstm(self.rnn)

    def forward(self, wd_emb, usr_emb, length):
        num_layers = self._num_layers
        batch_size = wd_emb.size()[0]
        output_size = self._output_size
        maxlenbatch = wd_emb.size()[1]
        #DBG HERE
        usr_emb = usr_emb.unsqueeze(1)
        usr_emb = usr_emb.expand(batch_size,maxlenbatch,usr_emb.size()[-1])
        #wd_emb = wd_emb.permute(1, 0, 2)
        #Concatenate these
        embed_seq =  T.cat((usr_emb, wd_emb),2)
        initial_state = (
                tovar(T.zeros(num_layers * 2, batch_size, output_size // 2)),
                tovar(T.zeros(num_layers * 2, batch_size, output_size // 2)),
                )
        #embed_seq: 93,160,26 length: 160,output_size: 18, init[0]: 2,160,9
        embed_seq = embed_seq.permute(1,0,2)
        embed, (h, c) = dynamic_rnn(self.rnn, embed_seq, length, initial_state)
        h = h.permute(1, 0, 2)
        return h[:, -2:].contiguous().view(batch_size, output_size)

class Context(NN.Module):
    def __init__(self,in_size, context_size, num_layers=1):
        NN.Module.__init__(self)
        self._context_size = context_size
        self._in_size = in_size
        self._num_layers = num_layers

        self.rnn = NN.LSTM(
                in_size,
                context_size,
                num_layers,
                bidirectional=False,
                )
        init_lstm(self.rnn)

    def forward(self, sent_encodings, length):
        num_layers = self._num_layers
        batch_size = sent_encodings.size()[0]
        context_size = self._context_size
        
        lstm_h = tovar(T.zeros(num_layers, batch_size, context_size))
        lstm_c = tovar(T.zeros(num_layers, batch_size, context_size))
        initial_state = (lstm_h, lstm_c)
        sent_encodings = sent_encodings.permute(1,0,2)
        embed, (h, c) = dynamic_rnn(self.rnn, sent_encodings, length, initial_state)
        embed = embed.contiguous().view(-1, context_size)
        #h = h.permute(1, 0, 2)
        return embed.view(batch_size, -1, context_size)



class Decoder(NN.Module):
    def __init__(self,size_usr, size_wd, context_size, num_words, 
                 state_size = None, num_layers=1):
        NN.Module.__init__(self)
        self._num_words = num_words
        self._context_size = context_size
        self._size_wd = size_wd
        self._size_usr = size_usr
        num_layers = 1 #NOT ALLOWED TO CHANGE NUM LAYERS
        self._num_layers = num_layers
        if state_size == None:
            state_size = size_usr + size_wd + context_size
        self._state_size = state_size

        self.rnn = NN.LSTM(
                size_wd + size_usr + context_size,
                state_size,
                num_layers,
                bidirectional=False,
                )
        self.softmax = HierarchicalLogSoftmax(state_size, np.int(np.sqrt(num_words)), num_words)
        init_lstm(self.rnn)

    def forward(self, context_encodings, wd_emb, wd_target, usr_emb, length):
        '''
        Returns:
            If wd_target is None, returns a 4D tensor P
                (batch_size, max_turns, max_sentence_length, num_words)
                where P[i, j, k, w] is the log probability of word w at sample i, dialogue j, word position k.
            If wd_target is a LongTensor (batch_size, max_turns, max_sentence_length), returns a tuple
                ((batch_size, max_turns, max_sentence_length), float)
                where the tensor contains the probability of ground truth (wd_target) and the float
                scalar is the log-likelihood.
        '''
        num_layers = self._num_layers
        batch_size = wd_emb.size()[0]
        maxlenbatch = wd_emb.size()[1]
        maxwordsmessage = wd_emb.size()[2]
        context_size = self._context_size
        size_wd = self._size_wd
        size_usr = self._size_usr
        state_size = self._state_size
        
        
        lstm_h = tovar(T.zeros(num_layers, batch_size * maxlenbatch, state_size))
        lstm_c = tovar(T.zeros(num_layers, batch_size * maxlenbatch, state_size))
        initial_state = (lstm_h, lstm_c)
        #batch, turns in a sample, words in a message, embedding_dim
        
        usr_emb = usr_emb.unsqueeze(2)
        usr_emb = usr_emb.expand(batch_size,maxlenbatch,maxwordsmessage,
                                 usr_emb.size()[-1])
        #length = length.unsqueeze(2)
        #length = length.expand(batch_size,maxlenbatch,maxwordsmessage,
        #                         length.size()[-1])
        context_encodings = context_encodings.unsqueeze(2)
        context_encodings = context_encodings.expand(
                batch_size,maxlenbatch,maxwordsmessage, context_encodings.size()[-1])
        
        embed_seq =  T.cat((usr_emb, wd_emb, context_encodings),3)
        embed_seq = embed_seq.view(batch_size * maxlenbatch, maxwordsmessage,-1)
        embed_seq = embed_seq.permute(1,0,2).contiguous()
        embed, (h, c) = dynamic_rnn(
                self.rnn, embed_seq, length.contiguous().view(-1),
                initial_state)
        h = h.permute(1, 0, 2)
        h = h[:, -1,:]
        embed = embed.permute(1, 0, 2).contiguous().view(batch_size, maxlenbatch, maxwordsmessage, state_size)
        if wd_target is None:
            out = self.softmax(embed.view(-1, state_size))
            out = out.view(batch_size, maxlenbatch, -1, self._num_words)
            return out#.contiguous().view(batch_size, maxlenbatch, maxwordsmessage, -1)
        else:
            target = T.cat((wd_target[:, :, 1:], tovar(T.zeros(batch_size, maxlenbatch, 1)).long()), 2)
            out = self.softmax(embed.view(-1, state_size), target.view(-1))
            out = out.view(batch_size, maxlenbatch, maxwordsmessage)
            mask = (target != 0).float()
            out = out * mask
            log_prob = out.sum() / mask.sum()
            return out, log_prob



dataset = UbuntuDialogDataset(
        'ubuntu', 
        'wordcount.pkl', 'usercount.pkl')

vcb = dataset.vocab
usrs = dataset.users
num_usrs = len(usrs)
vcb_len = len(vcb)
num_words = vcb_len
size_usr = 50
size_wd = 50
size_sentence = 300
size_context = 300

user_emb = cuda(NN.Embedding(num_usrs+1, size_usr, padding_idx = 0))
word_emb = cuda(NN.Embedding(vcb_len+1, size_wd, padding_idx = 0))
enc = cuda(Encoder(size_usr, size_wd, size_sentence, num_layers = 1))
context = cuda(Context(size_sentence, size_context, num_layers = 1))
decoder = cuda(Decoder(size_usr, size_wd, size_context, num_words+1))

params = sum([list(m.parameters()) for m in [user_emb, word_emb, enc, context, decoder]], [])
opt = T.optim.Adam(params, lr=1e-4)


dataloader = UbuntuDialogDataLoader(dataset, 1, num_workers=1)

for item in dataloader:
    turns, sentence_lengths_padded, speaker_padded, \
        addressee_padded, words_padded, words_reverse_padded = item
    words_padded = tovar(words_padded)
    words_reverse_padded = tovar(words_reverse_padded)
    speaker_padded = tovar(speaker_padded)
    addressee_padded = tovar(addressee_padded)
    sentence_lengths_padded = cuda(sentence_lengths_padded)
    turns = cuda(turns)
    
    batch_size = turns.size()[0]
    max_turns = words_padded.size()[1]
    max_words = words_padded.size()[2]
    #batch, turns in a sample, words in a message, embedding_dim
    wds_b = word_emb(words_padded.view(-1, max_words)).view(batch_size, max_turns, max_words, size_wd)
    wds_rev_b = word_emb(words_reverse_padded.view(-1, max_words)).view(batch_size, max_turns, max_words, size_wd)
    #batch, turns in a sample, embedding_dim
    usrs_b = user_emb(speaker_padded)
    addres_b = user_emb(addressee_padded)
    
    max_turns = turns.max()
    max_words = wds_b.size()[2]
    encodings = enc(wds_b.view(batch_size * max_turns, max_words, size_wd),
                usrs_b.view(batch_size * max_turns, size_usr), 
                sentence_lengths_padded.view(-1))
    encodings = encodings.view(batch_size, max_turns, -1)
    ctx = context(encodings, turns)
    max_output_words = sentence_lengths_padded[:, 1:].max()
    words_flat = words_padded[:,1:,:max_output_words].contiguous()
    # Training:
    prob, log_prob = decoder(ctx[:,:-1:], wds_b[:,1:,:max_output_words], words_flat,
                             usrs_b[:,1:], sentence_lengths_padded[:,1:])
    loss = -log_prob
    print(tonumpy(loss))
    opt.zero_grad()
    loss.backward()
    clip_grad(params, 1)
    opt.step()
    # Testing: during test time none of wds_b, ctx and sentence_lengths_padded is known.
    # We need to manually unroll the LSTMs.
    #decoded = decoder(ctx[:, :-1:], wds_b[:, 1:, :], None,
    #                  usrs_b[:, 1:,], sentence_lengths_padded[:, 1:])
    #print(decoded)



# Usage:
# dataset = UbuntuDialogDataset('../ubuntu-ranking-dataset-creator/src/dialogs', 'wordcount.pkl', 'usercount.pkl')
# dataloader = UbuntuDialogDataLoader(dataset, 16)
# for item in dataloader:
#     turns, sentence_lengths_padded, speaker_padded, addressee_padded, words_padded, words_reverse_padded = item
#     ...
#
# wordcount.pkl and usercount.pkl are generated from stage1 parser.
# If you want asynchronous data loading, use something like
# dataloader = UbuntuDialogDataLoader(dataset, 16, 4)
