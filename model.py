
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
        
        lstm_h = [tovar(T.zeros(batch_size, context_size)) for _ in range(num_layers)]
        lstm_c = [tovar(T.zeros(batch_size, context_size)) for _ in range(num_layers)]
        initial_state = [lstm_h, lstm_c]
        sent_encodings = sent_encodings.permute(1,0,2)
        embed, (h, c) = dynamic_rnn(self.rnn, sent_encodings, length, initial_state)
        
        h = h.permute(1, 0, 2)
        return h[:, -1,:].contiguous().view(batch_size, context_size)



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
        self.proj = NN.Sequential(
                NN.Linear(state_size, num_words),
                )
        self.softmax = NN.Softmax()
        init_weights(self.proj)
        init_lstm(self.rnn)

    def forward(self, context_encodings, wd_emb, usr_emb, length):
        num_layers = self._num_layers
        batch_size = context_encodings.size()[0]
        maxlenbatch = wd_emb.size()[1]
        maxwordsmessage = wd_emb.size()[2]
        context_size = self._context_size
        size_wd = self._size_wd
        size_usr = self._size_usr
        state_size = self._state_size
        
        
        lstm_h = [tovar(T.zeros(batch_size * maxlenbatch, state_size))
            for _ in range(num_layers)]
        lstm_c = [tovar(T.zeros(batch_size * maxlenbatch, state_size))
            for _ in range(num_layers)]
        initial_state = [lstm_h, lstm_c]
        #batch, turns in a sample, words in a message, embedding_dim
        
        usr_emb = usr_emb.unsqueeze(2)
        usr_emb = usr_emb.expand(batch_size,maxlenbatch,maxwordsmessage,
                                 usr_emb.size()[-1])
        #length = length.unsqueeze(2)
        #length = length.expand(batch_size,maxlenbatch,maxwordsmessage,
        #                         length.size()[-1])
        context_encodings = context_encodings.unsqueeze(1).unsqueeze(2)
        context_encodings = context_encodings.expand(
                batch_size,maxlenbatch,maxwordsmessage, context_encodings.size()[-1])
        
        embed_seq =  T.cat((usr_emb, wd_emb, context_encodings),3)
        embed_seq = embed_seq.view(batch_size * maxlenbatch, maxwordsmessage,-1)
        embed_seq = embed_seq.permute(1,0,2)
        embed, (h, c) = dynamic_rnn(
                self.rnn, embed_seq, length.view(-1), initial_state)
        h = h.permute(1, 0, 2)
        h = h[:, -1,:]
        out = self.proj(embed.contiguous().view(-1, state_size))
        out = self.softmax(out)
        out = out.view(batch_size, maxlenbatch, maxwordsmessage, -1)
        return out#.contiguous().view(batch_size, maxlenbatch, maxwordsmessage, -1)



dataset = UbuntuDialogDataset(
        'ubuntu', 
        'wordcount.pkl', 'usercount.pkl')

vcb = dataset.vocab
usrs = dataset.users
num_usrs = len(usrs)
vcb_len = len(vcb)
num_words = vcb_len
size_usr = 12
size_wd = 14
size_sentence = 18
size_context = 22

user_emb = NN.Embedding(num_usrs+1, size_usr, padding_idx = 0)
word_emb = NN.Embedding(vcb_len+1, size_wd, padding_idx = 0)
enc = Encoder(size_usr, size_wd, size_sentence, num_layers = 1)
context = Context(size_sentence, size_context, num_layers = 1)
decoder = Decoder(size_usr, size_wd, size_context, num_words+1)


dataloader = UbuntuDialogDataLoader(dataset, 2)

for item in dataloader:
    turns, sentence_lengths_padded, speaker_padded, \
        addressee_padded, words_padded, words_reverse_padded = item
    words_padded = T.autograd.Variable(words_padded)
    words_reverse_padded = T.autograd.Variable(words_reverse_padded)
    speaker_padded = T.autograd.Variable(speaker_padded)
    addressee_padded = T.autograd.Variable(addressee_padded)
    
    batch_size = turns.size()[0]
    #batch, turns in a sample, words in a message, embedding_dim
    wds_b = T.stack([word_emb(words_padded[i,:,:]) for i in range(batch_size)])
    wds_rev_b = T.stack([word_emb(words_reverse_padded[i,:,:]) for i in range(batch_size)])
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
    decoded = decoder(ctx, wds_b, usrs_b, sentence_lengths_padded)
    decoded_flat = decoded.view(batch_size * max_turns * max_words, -1)
    words_flat = words_padded.view(-1)
    perplex = decoded_flat.gather(1, words_flat.view(-1, 1))
    print(perplex)
    break



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
