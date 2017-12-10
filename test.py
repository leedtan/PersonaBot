
from torch.nn import Parameter
from functools import wraps
from nltk.translate import bleu_score
import nltk

import time
import torch as T
import torch.nn as NN
import torch.nn.functional as F
import torch.nn.init as INIT
import tensorflow as TF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy.random as RNG
import tensorflow as TF     # for Tensorboard
from numpy import rate


import argparse, sys, datetime, pickle, os

import matplotlib
matplotlib.use('Agg')
from mailbox import _create_carefully
import matplotlib.pyplot as PL

from PIL import Image


from torch.utils.data import DataLoader, Dataset
import numpy as np
np.set_printoptions(suppress=True)
from collections import Counter
from data_loader_stage1 import *

from adv import *
#from test import test


'''
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
'''
class Residual(NN.Module):
    def __init__(self,size, hidden):
        NN.Module.__init__(self)
        self.size = size
        self.linear1 = NN.Linear(size, hidden)
        self.linear2 = NN.Linear(hidden, size)
        self.relu = NN.LeakyReLU()

    def forward(self, x):
        h = self.relu(self.linear1(x))
        h = self.linear2(h)
        return self.relu(x + h)
class Dense(NN.Module):
    def __init__(self,size, hidden):
        NN.Module.__init__(self)
        self.size = size
        self.linear1 = NN.Linear(size, hidden)
        self.linear2 = NN.Linear(hidden, hidden)
        self.relu = NN.LeakyReLU()

    def forward(self, x):
        h = self.relu(self.linear1(x))
        h = self.linear2(h)
        return T.cat((x, self.relu(h)),1)


class Encoder(NN.Module):
    '''
    Inputs:
    @wd_emb: 3D (n_sentences, max_words, word_emb_size)
    @usr_emb: 2D (n_sentences, user_emb_size)
    @turns: 1D (n_sentences,) LongTensor

    Returns:
    @encoding: Sentence encoding,
        2D (n_sentences, output_size)
    @wds_h: Annotation vectors for each word
        3D (max_words, n_sentences, output_size)
    '''
    def __init__(self,size_usr, size_wd, output_size, num_layers, non_linearities=1):
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

    def forward(self, wd_emb, usr_emb, turns):
        wd_emb = cuda(wd_emb)
        usr_emb = cuda(usr_emb)
        turns = cuda(turns)

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
        embed, (h, c) = dynamic_rnn(self.rnn, embed_seq, turns, initial_state)
        h = h.permute(1, 0, 2)

        return h[:, -2:].contiguous().view(batch_size, output_size), embed

class Context(NN.Module):
    def __init__(self,in_size, context_size, size_attn, num_layers=1, non_linearities=1):
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
        self.attn_ctx = SelfAttention(size_context, size_context, size_attn, non_linearities = non_linearities)
        self.attn_wd = WdAttention(size_sentence, size_context, size_attn, non_linearities = non_linearities)
        self.attn_sent = RolledUpAttention(size_attn, size_context, size_attn, non_linearities = non_linearities)
        init_weights(self.attn_ctx)
        init_weights(self.attn_wd)
        init_weights(self.attn_sent)
        
        init_lstm(self.rnn)

    def zero_state(self, batch_size):
        lstm_h = tovar(T.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size))
        lstm_c = tovar(T.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size))
        initial_state = (lstm_h, lstm_c)
        return initial_state

    def forward(self, sent_encs, turns, sentence_lengths_padded,
                wds_h, usrs_b, initial_state=None):
            # attn: batch_size, max_turns, sentence_encoding_size
        sent_encs = cuda(sent_encs)
        turns = cuda(turns)
        initial_state = cuda(initial_state)

        num_layers = self._num_layers
        batch_size = sent_encs.size()[0]
        context_size = self._context_size
        
        if initial_state is None:
            initial_state = self.zero_state(batch_size)
        sent_encs = sent_encs.permute(1,0,2)
        embed, (h, c) = dynamic_rnn(self.rnn, sent_encs, turns, initial_state)
        embed = cuda(embed.permute(1,0,2).contiguous())

        # embed is now: batch_size, max_turns, context_size
        batch_size, num_turns, _ = embed.size()
        mask_size = [batch_size, num_turns, num_turns]
        ctx_mask = T.ones(*mask_size)
    
        # TODO describe @ctx_mask
        for i_b in range(mask_size[0]):
            for i_head in range(mask_size[1]):
                for i_sent in range(mask_size[2]):
                    if ((turns[i_b] <= i_head)
                        or (i_head < i_sent)
                        or (turns[i_b] <= i_sent)):
                            ctx_mask[i_b, i_head, i_sent] = 0
        ctx_mask = tovar(ctx_mask)
        ctx_attended = self.attn_ctx(embed, embed, ctx_mask)
        
        ctx = embed + ctx_attended
        #ctx = T.cat((embed, ctx_attended),2)
        
        wds_in_sample, _, size_sentence = wds_h.size()
        wds_h_attn = wds_h.view(wds_in_sample, batch_size, num_turns, 
                                size_sentence).permute(1,2,0,3).contiguous()
        
        size_wd_mask = [batch_size, num_turns, num_turns, wds_in_sample]
        wd_mask = T.ones(*size_wd_mask)

        # TODO describe @size_wd_mask
        for i_b in range(size_wd_mask[0]):
            for i_head in range(size_wd_mask[1]):
                for i_sent in range(size_wd_mask[2]):
                    for i_wd in range(size_wd_mask[3]):
                        if ((sentence_lengths_padded[i_b, i_sent] <= i_wd)
                                or (turns[i_b] <= i_head)
                                or (i_head < i_sent)
                                or (turns[i_b] <= i_sent)):
                            wd_mask[i_b, i_head, i_sent, i_wd] = 0
        wd_mask = tovar(wd_mask)
        # (batch_size, num_turns_attended_with, num_turns_attended_over, size_attn)
        wd_attended = self.attn_wd(wds_h_attn, ctx, wd_mask)
        
        # Use @ctx and @ctx_mask to attend over the attended words (TODO refine description.....)
        wd_sent_attended = self.attn_sent(wd_attended, ctx, ctx_mask)
        
        ctx = T.cat((ctx, wd_sent_attended),2)
        
        return ctx, h.contiguous()

def inverseHackTorch(tens):
    idx = [i for i in range(tens.size(2)-1,-1, -1)]
    idx = cuda(T.LongTensor(idx))
    inverted_tensor = tens[:,:,idx]
    return inverted_tensor

class Attention(NN.Module):
    '''
    Attention head: the one we are attending with
    Context: the one we are attending on
    '''
    def __init__(self, size_context, size_head, size_attn, num_layers = 1, non_linearities=1):
        NN.Module.__init__(self)
        self._size_context = size_context
        self._size_attn = size_attn
        self._num_layers = num_layers
        self.softmax = NN.Softmax()
        # Context projector
        if non_linearities == 1:
            self.F_ctx = NN.Sequential(
                NN.Linear(size_context, size_attn*2 * args.hidden_width),
                NN.LeakyReLU(),
                NN.Linear(size_attn*2 * args.hidden_width, size_attn))
            # Attendee projector
            self.F_head = NN.Sequential(
                NN.Linear(size_head, size_attn*2 * args.hidden_width),
                NN.LeakyReLU(),
                NN.Linear(size_attn*2 * args.hidden_width, size_attn))
            # Takes in context and attendee and produces a weight
            self.F_attn = NN.Sequential(
                NN.Linear(size_attn*2, size_attn * args.hidden_width),
                NN.LeakyReLU(),
                NN.Linear(size_attn * args.hidden_width, size_attn * args.hidden_width),
                NN.LeakyReLU(),
                NN.Linear(size_attn * args.hidden_width, 1))
        else:
            self.F_ctx = NN.Sequential(
                NN.Linear(size_context, size_attn))
            # Attendee projector
            self.F_head = NN.Sequential(
                NN.Linear(size_head, size_attn))
            # Takes in context and attendee and produces a weight
            self.F_attn = NN.Sequential(
                NN.Linear(size_attn*2, size_attn),
                NN.LeakyReLU(),
                NN.Linear(size_attn, 1))
            
        init_weights(self.F_head)
        init_weights(self.F_ctx)
        init_weights(self.F_attn)

class SelfAttention(Attention):
    '''
    Self-attention over historical sentences
    Input:
    @context: (batch_size, num_turns_context, size_context)
    @heads: (batch_size, num_turns_head, size_head)
    @mask: (batch_size, num_turns_head, num_turns_context)
    '''
    def __init__(self, size_context, size_head, size_attn, num_layers = 1, non_linearities=1):
        NN.Module.__init__(self)
        self._size_context = size_context
        self._size_attn = size_attn
        self._num_layers = num_layers
        self.softmax = NN.Softmax()
        # Context projector
        if non_linearities == 1:
            self.F_ctx = NN.Sequential(
                NN.Linear(size_context, size_attn*2 * args.hidden_width),
                NN.LeakyReLU(),
                NN.Linear(size_attn*2 * args.hidden_width, size_context))
            # Attendee projector
            self.F_head = NN.Sequential(
                NN.Linear(size_head, size_attn*2 * args.hidden_width),
                NN.LeakyReLU(),
                NN.Linear(size_attn*2 * args.hidden_width, size_attn))
            # Takes in context and attendee and produces a weight
            self.F_attn = NN.Sequential(
                NN.Linear(size_attn + size_context, size_attn * args.hidden_width),
                NN.LeakyReLU(),
                NN.Linear(size_attn * args.hidden_width, size_attn * args.hidden_width),
                NN.LeakyReLU(),
                NN.Linear(size_attn * args.hidden_width, 1))
        else:
            self.F_ctx = NN.Sequential(
                NN.Linear(size_context, size_context))
            # Attendee projector
            self.F_head = NN.Sequential(
                NN.Linear(size_head, size_attn))
            # Takes in context and attendee and produces a weight
            self.F_attn = NN.Sequential(
                NN.Linear(size_attn + size_context, size_attn),
                NN.LeakyReLU(),
                NN.Linear(size_attn, 1))
        init_weights(self.F_head)
        init_weights(self.F_ctx)
        init_weights(self.F_attn)
    
    def forward(self, context, heads, mask):
        batch_size, num_turns_ctx, size_context = context.size()
        _, num_turns_head, size_head = heads.size()
        size_attn = self._size_attn
        context = cuda(context)
        heads = cuda(heads)

        # Projection
        attn_ctx = context.view(batch_size * num_turns_ctx, size_context)
        attn_head = heads.view(batch_size * num_turns_head, size_head)
        attn_ctx_reduced = self.F_ctx(attn_ctx).view(batch_size, num_turns_ctx, size_context)
        attn_head_reduced = self.F_head(attn_head).view(batch_size, num_turns_head, size_attn)

        # Match all contexts and heads and compute the weights for every possible pairs
        # within a batch
        attn_ctx_expanded = attn_ctx_reduced.unsqueeze(1).expand(
            batch_size, num_turns_head, num_turns_ctx, size_context).contiguous().view(
                batch_size * num_turns_head * num_turns_ctx, size_context)
        attn_head_expanded = attn_head_reduced.unsqueeze(2).expand(
            batch_size, num_turns_head, num_turns_ctx, size_attn).contiguous().view(
                batch_size * num_turns_head * num_turns_ctx, size_attn)
        attn_raw = self.F_attn(T.cat((attn_head_expanded,attn_ctx_expanded),1)).view(
            batch_size, num_turns_head, num_turns_ctx)
        
        # Weighted average over heads
        attn_raw = weighted_softmax(
                attn_raw.view(batch_size * num_turns_head, num_turns_ctx),
                mask.view(batch_size*num_turns_head, num_turns_ctx)).view(
                        batch_size, num_turns_head, num_turns_ctx, 1)
        at_weighted_sent = attn_raw * attn_ctx_expanded.view(
                        batch_size, num_turns_head, num_turns_ctx, -1)
        at_weighted_sent = at_weighted_sent.sum(2)
        
        return at_weighted_sent

class WdAttention(Attention):
    '''
    @context: (batch_size, n_turns, max_words, size_sentence_encoding)
    @heads: (batch_size, n_turns, size_head)
    @mask: (batch_size, n_turns, n_turns, max_words)
    '''
    def forward(self, context, heads, mask):
        batch_size, num_turns_ctx, num_wds, size_context = context.size()
        _, num_turns_head, size_head = heads.size()

        size_attn = self._size_attn
        context = cuda(context)
        heads = cuda(heads)
        attn_ctx = context.view(batch_size * num_turns_ctx * num_wds, size_context)
        attn_head = heads.view(batch_size * num_turns_head, size_head)
        attn_ctx_reduced = self.F_ctx(attn_ctx).view(batch_size, num_turns_ctx * num_wds, size_attn)
        attn_head_reduced = self.F_head(attn_head).view(batch_size, num_turns_head, size_attn)

        attn_ctx_expanded = attn_ctx_reduced.unsqueeze(1).expand(
            batch_size, num_turns_head, num_turns_ctx * num_wds, size_attn).contiguous().view(
                batch_size * num_turns_head * num_turns_ctx * num_wds, size_attn)
        
        attn_head_expanded = attn_head_reduced.unsqueeze(2).expand(
            batch_size, num_turns_head, num_turns_ctx * num_wds, size_attn).contiguous().view(
                batch_size * num_turns_head * num_turns_ctx * num_wds, size_attn)
        
        attn_raw = self.F_attn(T.cat((attn_head_expanded,attn_ctx_expanded),1)).view(
            batch_size, num_turns_head, num_turns_ctx * num_wds)
        
        attn_raw = weighted_softmax(
                attn_raw.view(batch_size * num_turns_head * num_turns_ctx, num_wds),
                mask.view(batch_size*num_turns_head *num_turns_ctx,  num_wds)).view(
                        batch_size, num_turns_head, num_turns_ctx, num_wds, 1)
        at_weighted_sent = attn_raw * attn_ctx_expanded.view(
                        batch_size, num_turns_head, num_turns_ctx, num_wds, -1)
        at_weighted_sent = at_weighted_sent.sum(3)
        
        return at_weighted_sent

class RolledUpAttention(Attention):
    '''
    @context: (batch_size, num_turns_attended_with, num_turns_attended_over, size_attn)
    @head: (batch_size, num_turns_attended_with, size_head)
    @mask: (batch_size, num_turns_attended_with, num_turns_attended_over)
    '''
    def forward(self, context, heads, mask):
        batch_size, num_turns_head, num_turns_ctx, size_context = context.size()
        _, num_turns_head, size_head = heads.size()

        size_attn = self._size_attn
        context = cuda(context)
        heads = cuda(heads)

        attn_ctx = context.view(batch_size * num_turns_head * num_turns_ctx, size_context)
        attn_head = heads.view(batch_size * num_turns_head, size_head)
        attn_ctx_reduced = self.F_ctx(attn_ctx).view(batch_size, num_turns_head, num_turns_ctx, size_attn)
        attn_head_reduced = self.F_head(attn_head).view(batch_size, num_turns_head, size_attn)

        attn_ctx_expanded = attn_ctx_reduced.contiguous().view(
                batch_size * num_turns_head * num_turns_ctx, size_attn)
        
        attn_head_expanded = attn_head_reduced.unsqueeze(2).expand(
            batch_size, num_turns_head, num_turns_ctx, size_attn).contiguous().view(
                batch_size * num_turns_head * num_turns_ctx, size_attn)
        
        attn_raw = self.F_attn(T.cat((attn_head_expanded,attn_ctx_expanded),1)).view(
            batch_size, num_turns_head, num_turns_ctx)
        
        attn_raw = weighted_softmax(
                attn_raw.view(batch_size * num_turns_head, num_turns_ctx),
                mask.view(batch_size*num_turns_head, num_turns_ctx)).view(
                        batch_size, num_turns_head, num_turns_ctx, 1)
        at_weighted_sent = attn_raw * attn_ctx_expanded.view(
                        batch_size, num_turns_head, num_turns_ctx, -1)
        at_weighted_sent = at_weighted_sent.sum(2)
        
        return at_weighted_sent
    
class SeltAttentionWd(Attention):
    '''
    context: (batch_size, num_turns_ctx, max_words, size_context)
    head: (^, num_turns_head, ^, size_head)
    '''
    def forward(self, context, heads, mask):
        batch_size, num_turns_ctx, num_wds, size_context = context.size()
        _, _, _, size_head = heads.size()
        size_attn = self._size_attn
        context = cuda(context)
        heads = cuda(heads)
        attn_ctx = context.view(batch_size * num_turns_ctx * num_wds, size_context)
        attn_head = heads.view(batch_size * num_turns_ctx * num_wds, size_head)
        attn_ctx_reduced = self.F_ctx(attn_ctx).view(batch_size, num_turns_ctx,  num_wds, size_attn)
        attn_head_reduced = self.F_head(attn_head).view(batch_size, num_turns_ctx,  num_wds, size_attn)

        attn_ctx_expanded = attn_ctx_reduced.unsqueeze(2).expand(
            batch_size, num_turns_ctx, num_wds,  num_wds, size_attn).contiguous().view(
                batch_size * num_turns_ctx * num_wds * num_wds, size_attn)
        attn_head_expanded = attn_head_reduced.unsqueeze(3).expand(
            batch_size, num_turns_ctx, num_wds,  num_wds, size_attn).contiguous().view(
                batch_size * num_turns_ctx * num_wds * num_wds, size_attn)
        
        attn_raw = self.F_attn(T.cat((attn_head_expanded,attn_ctx_expanded),1)).view(
            batch_size, num_turns_ctx, num_wds, num_wds)
        
        attn_raw = weighted_softmax(
                attn_raw.view(batch_size * num_turns_ctx * num_wds, num_wds),
                mask.view(batch_size*num_turns_ctx * num_wds, num_wds)).view(
                        batch_size, num_turns_ctx, num_wds, num_wds, 1)
        at_weighted_sent = attn_raw * attn_ctx_expanded.view(
                        batch_size, num_turns_ctx, num_wds, num_wds, -1)
        at_weighted_sent = at_weighted_sent.sum(3)
        
        return at_weighted_sent
    
    def test(self,context, single_attn_head):
        #size head: 1, 
        batch_size, num_wds, size_context = context.size()
        batch_size, size_head = single_attn_head.size()
        size_attn = self._size_attn
        context = cuda(context)
        single_attn_head = cuda(single_attn_head)
        attn_ctx = context.view(batch_size * num_wds, size_context)
        attn_head = single_attn_head.view(batch_size, size_head)
        attn_ctx_reduced = self.F_ctx(attn_ctx)
        attn_head_reduced = self.F_head(attn_head).view(batch_size, size_attn)


        attn_head_expanded = attn_head_reduced.unsqueeze(1).expand(
            batch_size, num_wds, size_attn).contiguous().view(
                batch_size * num_wds, size_attn)
        
        attn_raw = self.F_attn(T.cat((attn_head_expanded,attn_ctx_reduced),1)).view(
            batch_size, num_wds)
        
        attn_raw = weighted_softmax(
                attn_raw.view(batch_size, num_wds)).view(
                        batch_size, num_wds,1)
        at_weighted_sent = attn_raw * attn_ctx_reduced.view(
                        batch_size, num_wds, -1)
        at_weighted_sent = at_weighted_sent.sum(1)
        
        return at_weighted_sent
        
class AttentionDecoderCtx(Attention):
    '''
    context: (batch_size, num_turns_ctx, size_context)
    head: (^, num_turns_head, num_wds, size_head)
    '''
    def forward(self, context, heads, mask):
        batch_size, num_turns_ctx, size_context = context.size()
        _, _, num_wds, size_head = heads.size()
        size_attn = self._size_attn
        context = cuda(context)
        heads = cuda(heads)
        attn_ctx = context.view(batch_size * num_turns_ctx, size_context)
        attn_head = heads.view(batch_size * num_turns_ctx * num_wds, size_head)
        attn_ctx_reduced = self.F_ctx(attn_ctx).view(batch_size, num_turns_ctx, size_attn)
        attn_head_reduced = self.F_head(attn_head).view(batch_size, num_turns_ctx,  num_wds, size_attn)

        attn_ctx_expanded = attn_ctx_reduced.unsqueeze(1).expand(
            batch_size, num_turns_ctx * num_wds, num_turns_ctx, size_attn).contiguous().view(
                batch_size, num_turns_ctx, num_wds, num_turns_ctx, size_attn).permute(0,1,3,2,4).contiguous().view(
                        batch_size * num_turns_ctx * num_turns_ctx * num_wds, size_attn)
        attn_head_expanded = attn_head_reduced.unsqueeze(2).expand(
            batch_size, num_turns_ctx, num_turns_ctx, num_wds, size_attn).contiguous().view(
                batch_size * num_turns_ctx * num_turns_ctx * num_wds, size_attn)
        
        attn_raw = self.F_attn(T.cat((attn_head_expanded,attn_ctx_expanded),1)).view(
            batch_size, num_turns_ctx, num_turns_ctx, num_wds)
        
        attn_raw = weighted_softmax(
                attn_raw.view(batch_size * num_turns_ctx * num_turns_ctx, num_wds),
                mask.view(batch_size*num_turns_ctx * num_turns_ctx, num_wds)).view(
                        batch_size, num_turns_ctx, num_turns_ctx, num_wds, 1)
        at_weighted_sent = attn_raw * attn_ctx_expanded.view(
                        batch_size, num_turns_ctx, num_turns_ctx, num_wds, -1)
        at_weighted_sent = at_weighted_sent.sum(2)
        
        return at_weighted_sent
    def test(self,context, heads, mask):
        num_turns_ctx, size_context = context.size()
        _, size_head = heads.size()
        size_attn = self._size_attn
        context = cuda(context)
        heads = cuda(heads)
        attn_ctx = context.view(num_turns_ctx, size_context)
        attn_head = heads.view(num_turns_ctx, size_head)
        attn_ctx_reduced = self.F_ctx(attn_ctx).view(num_turns_ctx, size_attn)
        attn_head_reduced = self.F_head(attn_head).view(num_turns_ctx, size_attn)

        attn_ctx_expanded = attn_ctx_reduced.unsqueeze(0).expand(
            num_turns_ctx, num_turns_ctx, size_attn).contiguous().view(
                        num_turns_ctx * num_turns_ctx, size_attn)
        attn_head_expanded = attn_head_reduced.unsqueeze(1).expand(
            num_turns_ctx, num_turns_ctx, size_attn).contiguous().view(
                num_turns_ctx * num_turns_ctx, size_attn)
        
        attn_raw = self.F_attn(T.cat((attn_head_expanded,attn_ctx_expanded),1)).view(
            num_turns_ctx, num_turns_ctx)
        
        attn_raw = weighted_softmax(
                attn_raw.view(num_turns_ctx, num_turns_ctx),
                mask.view(num_turns_ctx, num_turns_ctx)).view(
                        num_turns_ctx, num_turns_ctx, 1)
        at_weighted_sent = attn_raw * attn_ctx_expanded.view(
                        num_turns_ctx, num_turns_ctx, -1)
        at_weighted_sent = at_weighted_sent.sum(1)
        
        return at_weighted_sent
        
class Decoder(NN.Module):
    def __init__(self,size_usr, size_wd, context_size, size_sentence, size_attn, num_words, max_len_generated ,beam_size,
                 state_size = None, num_layers=1, non_linearities=1):
        NN.Module.__init__(self)
        self._num_words = num_words
        self._beam_size = beam_size
        self._max_len_generated = max_len_generated
        self._context_size = context_size
        self._size_wd = size_wd
        self._size_usr = size_usr
        self._num_layers = num_layers
#       in_size = size_usr + size_wd + context_size + size_attn
        init_size = size_usr + context_size + size_attn
        in_size = size_wd
        RNN_in_size = in_size//2
        self._RNN_in_size = RNN_in_size
        if state_size == None:
            state_size = in_size
        self._state_size = state_size
        decoder_out_size = state_size + size_attn*2
        f_out_size = decoder_out_size + size_wd
        if non_linearities == 1:
            self.F_init_h = NN.Sequential(
                    NN.Linear(init_size, state_size * num_layers * args.hidden_width),
                    NN.LeakyReLU(),
                    NN.Linear(state_size * num_layers * args.hidden_width, state_size * num_layers),
                    NN.Tanh()
                    )
            self.F_init_c = NN.Sequential(
                    NN.Linear(init_size, state_size * num_layers * args.hidden_width),
                    NN.LeakyReLU(),
                    NN.Linear(state_size * num_layers * args.hidden_width, state_size * num_layers)
                    )
            self.F_reconstruct = NN.Sequential(
                NN.Linear(decoder_out_size, decoder_out_size//2 * args.hidden_width),
                NN.LeakyReLU(),
                NN.Linear(decoder_out_size//2 * args.hidden_width, size_wd)
                )
        else:
            self.F_init_h = NN.Sequential(
                    NN.Linear(init_size,  state_size * num_layers),
                    NN.Tanh()
                    )
            self.F_init_c = NN.Sequential(
                    NN.Linear(init_size,  state_size * num_layers)
                    )
            self.F_reconstruct = NN.Sequential(
                NN.Linear(decoder_out_size, decoder_out_size//2),
                NN.LeakyReLU(),
                NN.Linear(decoder_out_size//2, size_wd)
                )
        '''
        self.F_in = NN.Sequential(
            NN.Linear(in_size, in_size//2),
            NN.LeakyReLU(),
            NN.Linear(in_size//2, RNN_in_size)
            )
        init_weights(self.F_in)
        '''
        
        self.F_output = NN.Sequential(
            Residual(f_out_size, f_out_size//2),
            Dense(f_out_size, f_out_size),
            Residual(f_out_size*2, f_out_size),
            Dense(f_out_size*2, f_out_size),
            NN.Linear(f_out_size * 3, decoder_out_size)
            )
        self.rnn = NN.LSTM(
                in_size + 1,
                state_size,
                num_layers,
                bidirectional=False,
                )
        init_weights(self.F_reconstruct)
        init_weights(self.F_output)
        self.softmax = HierarchicalLogSoftmax(decoder_out_size, np.int(np.sqrt(num_words)), num_words)
        init_lstm(self.rnn)
        self.SeltAttentionWd = SeltAttentionWd(state_size + size_wd, state_size, size_attn,
                                               non_linearities = non_linearities)
        self.AttentionDecoderCtx = AttentionDecoderCtx(
                size_context + size_attn + size_usr, state_size + size_attn, size_attn,
                non_linearities = non_linearities)
        init_weights(self.SeltAttentionWd)
        init_weights(self.AttentionDecoderCtx)

    def zero_state(self, batch_size, ctx):
        lstm_h = self.F_init_h(ctx.view(batch_size, -1))
        lstm_h = lstm_h.view(batch_size, self._num_layers, self._state_size).permute(1, 0, 2)
        lstm_c = self.F_init_c(ctx.view(batch_size, -1))
        lstm_c = lstm_c.view(batch_size, self._num_layers, self._state_size).permute(1, 0, 2)
        initial_state = (lstm_h.contiguous(), lstm_c.contiguous())
        return initial_state

    def forward(self, context_encodings, wd_emb, usr_emb, sentence_lengths_padded, 
                wd_target=None, initial_state=None, wds_b_reconstruct = None):
        '''
        Returns:
            If wd_target is None, returns a 4D tensor P
                (batch_size, max_turns, max_sentence_length, num_words)
                where P[i, j, k, w] is the log probability of word w at sample i, utterance j, word position k.
            If wd_target is a LongTensor (batch_size, max_turns, max_sentence_length), returns a tuple
                ((batch_size, max_turns, max_sentence_length), float)
                where the tensor contains the probability of ground truth (wd_target) and the float
                scalar is the log-likelihood.
        '''
        context_encodings = cuda(context_encodings)
        wd_emb = cuda(wd_emb)
        usr_emb = cuda(usr_emb)
        sentence_lengths_padded = cuda(sentence_lengths_padded)
        wd_target = cuda(wd_target)
        initial_state = cuda(initial_state)

        batch_size, maxlenbatch, maxwordsmessage, _ = wd_emb.size()
        num_turns = maxlenbatch
        state_size = self._state_size
            
        ctx_for_attn = T.cat((context_encodings, usr_emb),2)    
        if initial_state is None:
            initial_state = self.zero_state(
                    batch_size * maxlenbatch, ctx_for_attn)
        
        #batch, turns in a sample, words in a message, embedding_dim

        usr_emb = usr_emb.unsqueeze(2)
        usr_emb = usr_emb.expand(batch_size,maxlenbatch,maxwordsmessage,
                                 usr_emb.size()[-1])
        
        
        
        context_encodings = context_encodings.unsqueeze(2)
        context_encodings = context_encodings.expand(
                batch_size,maxlenbatch,maxwordsmessage, context_encodings.size()[-1])
        self.context_encodings = context_encodings

        #embed_seq =  T.cat((usr_emb, wd_emb, context_encodings),3)
        indexes_in_sent = tovar(T.arange(0,maxwordsmessage).unsqueeze(0).unsqueeze(0).unsqueeze(3).expand(
                batch_size, maxlenbatch, maxwordsmessage,1
                ))
        
        embed_seq = T.cat((wd_emb, indexes_in_sent),3).contiguous()
        '''
        embed_seq = self.F_in(embed_seq.view(batch_size * maxlenbatch * maxwordsmessage,-1))
        '''
        embed_seq = embed_seq.view(batch_size * maxlenbatch, maxwordsmessage,-1)
        embed_seq = embed_seq.permute(1,0,2).contiguous()
        embed, (h, c) = dynamic_rnn(
                self.rnn, embed_seq, sentence_lengths_padded.contiguous().view(-1),
                initial_state)
        self.rnn_output = embed
        self.rnn_state = (h, c)
        maxwordsmessage = embed.size()[0]
        embed = embed.permute(1, 0, 2).contiguous().view(batch_size, maxlenbatch, maxwordsmessage, -1)
        wd_emb_attn = wd_emb[:,:,:maxwordsmessage,:]
        embed_shifted = T.cat((tovar(T.zeros((batch_size, maxlenbatch, 1, state_size))), embed[:,:,:-1,:]),2)
        embed_attn = T.cat((embed_shifted, wd_emb_attn),3)
        self.embed_attn = embed_attn

        # TODO: describe @size_wd_mask
        size_wd_mask = [batch_size, num_turns, maxwordsmessage, maxwordsmessage]
        wd_mask = T.ones(*size_wd_mask)

        for i_b in range(size_wd_mask[0]):
            for i_sent in range(size_wd_mask[1]):
                for i_wd_head in range(size_wd_mask[2]):
                    for i_wd_ctx in range(size_wd_mask[3]):
                        if ((sentence_lengths_padded[i_b, i_sent] <= i_wd_ctx)
                                or (turns[i_b] <= i_sent)
                                or (i_wd_ctx >= i_wd_head)):
                            wd_mask[i_b, i_sent, i_wd_head, i_wd_ctx] = 0
        wd_mask = tovar(wd_mask)
        
        
        attn = self.SeltAttentionWd(embed_attn, embed, wd_mask)
        self.attn = attn
        
        
        size_ctx_mask = [batch_size, num_turns, num_turns, maxwordsmessage]
        ctx_mask = T.ones(*size_ctx_mask)

        for i_b in range(size_ctx_mask[0]):
            for i_ctx in range(size_ctx_mask[1]):
                for i_head in range(size_ctx_mask[2]):
                    if ((turns[i_b] <= i_ctx)
                        or (turns[i_b] <= i_head)
                        or (i_ctx > i_head)):
                            ctx_mask[i_b, i_ctx, i_head, :] = 0
        ctx_mask = tovar(ctx_mask)
        
        embed = T.cat((embed, attn),3)
        attn = self.AttentionDecoderCtx(ctx_for_attn, embed, ctx_mask)
        self.attn2 = attn
        embed = T.cat((embed, attn),3)
        embed = embed.view(-1, state_size + size_attn * 2)
        reconstruct = self.F_reconstruct(embed)
        embed = T.cat((embed, reconstruct),1)
        embed = self.F_output(embed)
        
        if wd_target is None:
            out = self.softmax(embed)
            out = out.view(batch_size, maxlenbatch, -1, self._num_words)
            log_prob = None
        else:
            target = T.cat((wd_target[:, :, 1:], tovar(T.zeros(batch_size, maxlenbatch, 1)).long()), 2)
            decoder_out = embed
            out = self.softmax(decoder_out, target.view(-1))
            out = out.view(batch_size, maxlenbatch, maxwordsmessage)
            mask = (target != 0).float()
            out = out * mask
            log_prob = out.sum() / mask.sum()
            if wds_b_reconstruct is not None:
                
                reconstruct_loss = ((reconstruct.view(
                        batch_size, maxlenbatch, maxwordsmessage, size_wd)
                            - wds_b_reconstruct.detach()) ** 2) * mask.unsqueeze(-1)
                reconstruct_loss_mean = reconstruct_loss.sum() / mask.sum()

        if log_prob is None:
            return out, (h, c)#.contiguous().view(batch_size, maxlenbatch, maxwordsmessage, -1)
        else:
            if wds_b_reconstruct is None:
                return out, log_prob, (h, c),
            else:
                return out, log_prob, (h, c), reconstruct_loss_mean

    def get_next_word(self, prev_word, wd_emb_history_for_attn, rnn_output_history,
                      ctx_history_for_attn, cur_state, Bleu = False):
        """
        :param embed_seq: max_words, num_sentences_decoding, state_size
        :param wd_emb_for_attention: max_words, num_sentences_decoding, size_wd
        :cur_state: current state of decoder RNN
        :rnn_output_history: decoder's previous RNN states
            num_sentences_decoding, max_words - 1, state_size or None
        """
        global sentence_lengths_padded, turns
        prev_word = cuda(prev_word)
        wd_emb_history_for_attn = cuda(wd_emb_history_for_attn)
        ctx_history_for_attn = cuda(ctx_history_for_attn)
        cur_state = cuda(cur_state)
        state_size = self._state_size

        num_sentences_parallel, num_wds_so_far, state_size_seq = wd_emb_history_for_attn.size()
        '''
        embed_seq = self.F_in(embed_seq.view(num_wds * num_decoded, state_size_seq)).view(
                num_wds, num_decoded, -1)
        '''
        indexes_in_sent = tovar(np.tile(num_wds_so_far, num_sentences_parallel)).view(num_sentences_parallel,1).float()
        
        rnn_input = T.cat((prev_word, indexes_in_sent),1).contiguous()
        rnn_input = rnn_input.view(
                1,num_sentences_parallel, -1)
        rnn_output, current_state = self.rnn(rnn_input, cur_state)
        rnn_output = rnn_output.squeeze().contiguous()
        #number sentences parallel, num_words, size_emb
        attn_ctx = T.cat((rnn_output_history, wd_emb_history_for_attn),2)
        
        attn = self.SeltAttentionWd.test(
                attn_ctx, rnn_output)
        #attn is num_sentences_parallel by attn_ctx
        
        embed = T.cat((rnn_output, attn),1)  
        
        size_ctx_mask = [num_sentences_parallel, num_sentences_parallel]
        ctx_mask = T.ones(*size_ctx_mask)
        for i_ctx in range(size_ctx_mask[0]):
            for i_head in range(size_ctx_mask[1]):
                if (i_ctx > i_head):
                        ctx_mask[i_ctx, i_head] = 0
        ctx_mask = tovar(ctx_mask)
        attn = self.AttentionDecoderCtx.test(ctx_history_for_attn, embed, ctx_mask)
        #attn is num_sentences_parallel by ctx_history_for_attn[-1]
        embed = T.cat((embed, attn),1)
        
        #embed = embed.view(batch_size, -1, maxwordsmessage, self._state_size*2)
        embed = embed.view(num_sentences_parallel, state_size + size_attn + size_attn).contiguous()
        reconstruct = self.F_reconstruct(embed)
        embed = T.cat((embed, reconstruct),1)
        embed = self.F_output(embed)
        out = self.softmax(embed)
        #out = gaussian(out, True, 0, 10/(1+np.sqrt(itr)))
        if Bleu:
            indexes = out.exp().multinomial().detach()
            logp_selected = out.gather(1, indexes)
            return indexes, current_state, rnn_output, logp_selected
        else:
            indexes = out.topk(1, 1)[1]
            return indexes, current_state, rnn_output, False

    def greedyGenerateBleu(self, context_encodings, usr_emb, word_emb, dataset, Bleu=True):
        """
        How to require_grad=False ?
        :param context_encodings: (batch_size x context_size)
        :param word_emb:  idx to vector word embedder.
        :param usr_emb: (batch_size x usr_emb_size)
        :return: response : (batch_size x max_response_length)
        """
        context_encodings = cuda(context_encodings)
        usr_emb = cuda(usr_emb)
        word_emb = cuda(word_emb)

        num_layers = self._num_layers
        state_size = self._state_size
        max_len_generated = self._max_len_generated

        batch_size = context_encodings.size(0)
        ctx_for_attn = T.cat((context_encodings, usr_emb),1)
        cur_state = self.zero_state(batch_size, ctx_for_attn)

        # Initial word of response : Start token
        init_word = tovar(T.LongTensor(batch_size).fill_(dataset.index_word(START)))
        # End of generated sentence : EOS token
        stop_word = cuda(T.LongTensor(batch_size).fill_(dataset.index_word(EOS)))

        current_w = init_word
        output = tovar(current_w.data.unsqueeze(1))
        logprob = None
        init_seq = 0
        while not stop_word.equal(current_w.data.squeeze()) and output.size(1) < max_len_generated:
            current_w_emb = word_emb(current_w.squeeze())
            if init_seq == 0:
                init_seq = 1
                wd_emb_for_attn = current_w_emb.unsqueeze(1).contiguous()
                rnn_outputs = tovar(T.zeros(wd_emb_for_attn.size()[0], 1, state_size))
            else:
                rnn_outputs = T.cat((rnn_outputs, rnn_output.unsqueeze(1).contiguous()),1)
                wd_emb_for_attn = T.cat((wd_emb_for_attn, current_w_emb.unsqueeze(1).contiguous()),1)
            
            current_w, cur_state, rnn_output, current_logprob = self.get_next_word(
                    current_w_emb, wd_emb_for_attn, rnn_outputs,
                      ctx_for_attn, cur_state, Bleu = Bleu)
            output = T.cat((output, current_w), 1)
            if Bleu:
                logprob = T.cat((logprob, current_logprob), 1) if logprob is not None else current_logprob
            
        return output, logprob

parser = argparse.ArgumentParser(description='Ubuntu Dialogue dataset parser')
parser.add_argument('--dataroot', type=str,default='ubuntu', help='Root of the data downloaded from github')
parser.add_argument('--metaroot', type=str, default='ubuntu-meta', help='Root of meta data')
parser.add_argument('--vocabsize', type=int, default=39996, help='Vocabulary size')
parser.add_argument('--gloveroot', type=str,default='glove', help='Root of the data downloaded from github')
parser.add_argument('--outputdir', type=str, default ='outputs',help='output directory')
parser.add_argument('--logdir', type=str, default='logs', help='log directory')
parser.add_argument('--encoder_layers', type=int, default=3)
parser.add_argument('--decoder_layers', type=int, default=1)
parser.add_argument('--context_layers', type=int, default=1)
parser.add_argument('--size_context', type=int, default=256)
parser.add_argument('--size_sentence', type=int, default=128)
parser.add_argument('--size_attn', type=int, default=64)
parser.add_argument('--decoder_size_sentence', type=int, default=512)
parser.add_argument('--decoder_beam_size', type=int, default=4)
parser.add_argument('--decoder_max_generated', type=int, default=30)
parser.add_argument('--size_usr', type=int, default=16)
parser.add_argument('--size_wd', type=int, default=50)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--gradclip', type=float, default=1)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--modelname', type=str, default = '')
parser.add_argument('--modelnamesave', type=str, default='')
parser.add_argument('--modelnameload', type=str, default='')
parser.add_argument('--loaditerations', type=int, default=0)
parser.add_argument('--max_sentence_length_allowed', type=int, default=30)
parser.add_argument('--max_turns_allowed', type=int, default=8)
parser.add_argument('--num_loader_workers', type=int, default=4)
parser.add_argument('--adversarial_sample', type=int, default=1)
parser.add_argument('--emb_gpu_id', type=int, default=0)
parser.add_argument('--ctx_gpu_id', type=int, default=0)
parser.add_argument('--enc_gpu_id', type=int, default=0)
parser.add_argument('--dec_gpu_id', type=int, default=0)
parser.add_argument('--lambda_pg', type=float, default=.01)
parser.add_argument('--lambda_repetitive', type=float, default=.3)
parser.add_argument('--lambda_reconstruct', type=float, default=.1)
parser.add_argument('--non_linearities', type=int, default=1)
parser.add_argument('--hidden_width', type=int, default=1)
parser.add_argument('--server', type=int, default=0)

args = parser.parse_args()
if args.server == 1:
    args.dataroot = '/misc/vlgscratch4/ChoGroup/gq/data/OpenSubtitles/OpenSubtitles-dialogs/'
    args.metaroot = 'opensub'
    args.logdir = '/home/qg323/lee/'
print(args)

datasets = []
dataloaders = []
subdirs = []
for subdir in os.listdir(args.dataroot):
    print('Loading dataset:', subdir)
    dataset = UbuntuDialogDataset(os.path.join(args.dataroot, subdir),
                                  wordcount_pkl=args.metaroot + '/wordcount.pkl',
                                  usercount_pkl=args.metaroot + '/usercount.pkl',
                                  turncount_pkl=args.metaroot + '/turncount.pkl',
                                  max_sentence_lengths_pkl=args.metaroot + '/max_sentence_lengths.pkl',
                                  max_sentence_length_allowed=args.max_sentence_length_allowed,
                                  max_turns_allowed=args.max_turns_allowed,
                                  vocab_size=args.vocabsize)
    datasets.append(dataset)
    # Note that all datasets share the same vocabulary, users, and all the metadatas.
    # The only difference between datasets are the samples.
    dataloader = UbuntuDialogDataLoader(dataset, args.batchsize, num_workers=args.num_loader_workers)
    dataloaders.append(dataloader)
    subdirs.append(subdir)

print('Checking consistency...')
for dataset in datasets:
    assert all(w1 == w2 for w1, w2 in zip(datasets[0].vocab, dataset.vocab))
    assert all(u1 == u2 for u1, u2 in zip(datasets[0].users, dataset.users))


extra_penalty = np.zeros(args.max_sentence_length_allowed+1)
extra_penalty[0] = 5
extra_penalty[1] = 4
extra_penalty[2] = 3
extra_penalty[3] = 2


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

    user_emb_filename_base = '%s-user_emb' % args.modelnameload
    dirname = os.path.dirname(user_emb_filename_base)
    if dirname == '':
        dirname = None
    filename_base = os.path.basename(user_emb_filename_base)
    candidates = [name for name in os.listdir(dirname) if name.startswith(filename_base)]
    latest_time = 0
    for cand in candidates:
        mtime = os.path.getmtime(cand)
        try:
            loaditer = int(cand[-8:])
        except ValueError:
            print('Skipping %s' % cand)
            continue
        if mtime > latest_time:
            latest_time = mtime
            latest_loaditer = loaditer

    args.loaditerations = latest_loaditer


def logdirs(logdir, modelnamesave):
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    logdir = (
            logdir + modelnamesave
            )
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    elif not os.path.isdir(logdir):
        raise IOError('%s is not a directory' % logdir)
    return logdir
log_train = logdirs(args.logdir, modelnamesave)

train_writer = TF.summary.FileWriter(log_train)

adv_min_itr = 1000


vcb = dataset.vocab
usrs = dataset.users
num_usrs = len(usrs)
vcb_len = len(vcb)
num_words = vcb_len
size_usr = args.size_usr
size_wd = args.size_wd
size_sentence = args.size_sentence
size_context = args.size_context
size_attn = args.size_attn = 10 #For now hard coding to 10.
decoder_size_sentence = args.decoder_size_sentence
decoder_beam_size = args.decoder_beam_size
decoder_max_generated = args.decoder_max_generated

user_emb = cuda(NN.Embedding(num_usrs+1, size_usr, padding_idx = 0, scale_grad_by_freq=True))
word_emb = cuda(NN.Embedding(vcb_len+1, size_wd, padding_idx = 0, scale_grad_by_freq=True))
# If you want to preprocess other glove embeddings.
# preprocess_glove(args.gloveroot, 100)
word_emb.weight.data.copy_(init_glove(word_emb, vcb, dataset._ivocab, args.gloveroot))
enc = cuda(Encoder(size_usr, size_wd, size_sentence, num_layers = args.encoder_layers,
                   non_linearities = args.non_linearities))
context = cuda(Context(size_sentence, size_context, size_attn,
               num_layers = args.context_layers, non_linearities = args.non_linearities))
decoder = cuda(Decoder(size_usr, size_wd, size_context, size_sentence, size_attn, num_words+1,
               decoder_max_generated, decoder_beam_size, 
               state_size=decoder_size_sentence, 
               num_layers = args.decoder_layers, non_linearities = args.non_linearities))

eos = dataset.index_word(EOS)

loss_nan = reg_nan = grad_nan = 0

itr = args.loaditerations
epoch = 0
usr_std = wd_std = sent_std = ctx_std = wds_h_std = 1e-7
adv_emb_diffs = []
adv_sent_diffs = []
adv_ctx_diffs = []
adv_emb_scales = []
adv_sent_scales = []
adv_ctx_scales = []
baseline = None
if modelnameload:
    if len(modelnameload) > 0:
        user_emb = T.load('%s-user_emb-%08d' % (modelnameload, args.loaditerations))
        word_emb = T.load('%s-word_emb-%08d' % (modelnameload, args.loaditerations))
        enc = T.load('%s-enc-%08d' % (modelnameload, args.loaditerations))
        context = T.load('%s-context-%08d' % (modelnameload, args.loaditerations))
        decoder = T.load('%s-decoder-%08d' % (modelnameload, args.loaditerations))

params = sum([list(m.parameters()) for m in [user_emb, word_emb, enc, context, decoder]], [])
named_params = sum([list(m.named_parameters())
    for m in [user_emb, word_emb, enc, context, decoder]], [])

def enable_train(sub_modules):
    for m in sub_modules:
        m.train()
def enable_eval(sub_modules):
    for m in sub_modules:
        m.eval()
opt = T.optim.Adam(params, lr=args.lr)
#opt = T.optim.RMSprop(params, lr=args.lr,weight_decay=1e-6)

for subdir, dataloader in zip(subdir, dataloaders):
    adv_style = 0
    scatter_entropy_freq = 200
    time_train = 0
    time_decode = 0
    epoch += 1

    sum_ppl = 0
    count_ppl = 0
    ngram_counts = [Counter() for _ in range(3)]
    for item in dataloader:
        turns, sentence_lengths_padded, speaker_padded, \
            addressee_padded, words_padded, words_reverse_padded = item
        wds = sentence_lengths_padded.sum()
        max_wds = args.max_turns_allowed * args.max_sentence_length_allowed
        if sentence_lengths_padded.size(1) < 2:
            continue
        if wds > max_wds * .8 or wds < max_wds * .05:
            continue
        words_padded = tovar(words_padded)
        words_reverse_padded = tovar(words_reverse_padded)
        speaker_padded = tovar(speaker_padded)
        
        #enable_train([user_emb, word_emb, enc, context, decoder])
        #SO far not used
        #addressee_padded = tovar(addressee_padded)
        #addres_b = user_emb(addressee_padded)
        sentence_lengths_padded = cuda(sentence_lengths_padded)
        turns = cuda(turns)
        
        batch_size = turns.size()[0]
        max_turns = words_padded.size()[1]
        max_words = words_padded.size()[2]
        #batch, turns in a sample, words in a message, embedding_dim
        wds_b = word_emb(words_padded.view(-1, max_words)).view(batch_size, max_turns, max_words, size_wd)
        #SO FAR NOT USED:
        #wds_rev_b = word_emb(words_reverse_padded.view(-1, max_words)).view(batch_size, max_turns, max_words, size_wd)
        #batch, turns in a sample, embedding_dim
        usrs_b = user_emb(speaker_padded)

        # Not sure why these two statements are here...
        max_turns = turns.max()
        max_words = wds_b.size()[2]

        # Get the encoding vector for each sentence (@encodings), as well as
        # annotation vectors for each word in each sentence (@wds_h) so that we
        # can attend later.
        # The encoder (@enc) takes in a batch of word embedding seqences, a batch of
        # users, and a batch of sentence lengths.
        encodings, wds_h = enc(wds_b.view(batch_size * max_turns, max_words, size_wd),usrs_b.view(
                batch_size * max_turns, size_usr), sentence_lengths_padded.view(-1))

        assert np.all(~np.isnan(tonumpy(encodings)))
        assert np.all(~np.isnan(tonumpy(wds_h)))
        # Do the same for word-embedding, user-embedding, and encoder network
        encodings = encodings.view(batch_size, max_turns, -1)
        ctx, _ = context(encodings, turns, sentence_lengths_padded, wds_h.contiguous(), usrs_b)

        # Do the same for everything except the decoder
        max_output_words = sentence_lengths_padded[:, 1:].max()
        wds_b_decode = wds_b[:,1:,:max_output_words]
        wds_b_reconstruct = T.cat((
                wds_b_decode[:,:,1:,:], cuda(T.zeros(wds_b_decode.size()[0], 
                wds_b_decode.size()[1], 1, wds_b_decode.size()[3]))),2).contiguous()
        usrs_b_decode = usrs_b[:,1:]
        words_flat = words_padded[:,1:,:max_output_words].contiguous()
        prob, log_prob, _, reconstruct_loss_mean = decoder(ctx[:,:-1:], wds_b_decode,
                                 usrs_b_decode, sentence_lengths_padded[:,1:], 
                                 words_flat, wds_b_reconstruct = wds_b_reconstruct)
        loss = -log_prob

        #print(loss, grad_norm)
        '''
        if itr % 100 == 0:
            print('loss_nan', loss_nan, 'reg_nan', reg_nan, 'grad_nan', grad_nan)
        if np.any(np.isnan(tonumpy(loss))):
            loss_nan = 1
            print('LOSS NAN')
            raise ValueError
            continue
        if np.any(np.isnan(tonumpy(reg))):
            reg_nan = 1
            print('REG NAN')
            raise ValueError
            continue
        if np.any(np.isnan(tonumpy(grad_norm))):
            grad_nan = 1
            print('grad_norm NAN')
            raise ValueError
            continue
        #print('Grad norm', grad_norm)
        '''

        sum_ppl += np.exp(np.asscalar(tonumpy(loss)))
        count_ppl += 1

        # Train with Policy Gradient on BLEU scores once for a while.
        start_decode = time.time()
        #enable_eval([user_emb, word_emb, enc, context, decoder])
        greedy_responses, _ = decoder.greedyGenerateBleu(
                ctx[:1,:-1,:].view(-1, size_context + size_attn),
                  usrs_b_decode[:1,:,:].view(-1, size_usr), word_emb, dataset)
        # Only take the first turns[0] responses
        greedy_responses = greedy_responses[:turns[0]]
        reference = tonumpy(words_padded[0,1:turns[0],:])
        hypothesis = tonumpy(greedy_responses)
        
        # Compute BLEU scores
        real_sent = []
        gen_sent = []
        BLEUscores = []
        BLEUscoresplot = []
        lengths_gen = []
        batch_words = Counter()
        smoother = bleu_score.SmoothingFunction()
        for idx in range(reference.shape[0]):
            real_sent.append(reference[idx, :sentence_lengths_padded[0,idx]])
            num_words = np.where(hypothesis[idx,:]==eos)[0]
            if len(num_words) < 1:
                num_words = hypothesis.shape[1]
            else:
                num_words = num_words[0]
            lengths_gen.append(num_words)
            gen_sent.append(hypothesis[idx, :num_words+1])
            batch_words.update(gen_sent[-1][1:])
            curr_bleu = bleu_score.sentence_bleu(
                    [real_sent[-1]], gen_sent[-1], smoothing_function=smoother.method1)
            BLEUscoresplot.append(curr_bleu)
            curr_bleu += num_words / (1+np.sqrt(itr))
            
            #curr_bleu += extra_penalty[num_words]/(1+np.sqrt(itr))
            BLEUscores.append(curr_bleu)
        
        '''
        for idx in range(reference.shape[0]):
            greedy_responses[idx,:num_words].reinforce(BLEUscores[idx] - baseline)
        '''

        # Dump...
        #print('REAL:',dataset.translate_item(None, None, tonumpy(words_padded[:1,:,:])))
        greedy_responses = tonumpy(greedy_responses)
        
        words_padded_decode = tonumpy(words_padded[0,:,:])
        for i in range(greedy_responses.shape[0]):
                
            end_idx = np.where(words_padded_decode[i,:]==eos)
            printed = 0
            if len(end_idx) > 0:
                end_idx = end_idx[0]
                if len(end_idx) > 0:
                    end_idx = end_idx[0]
                    if end_idx > 0:
                        speaker, _, words = dataset.translate_item(tonumpy(speaker_padded[0:1, i]), None, words_padded_decode[i:i+1,:end_idx+1])
                        print('Real:', speaker[0], ' '.join(words[0]))
                        printed = 1
            if printed == 0 and words_padded_decode[i, 1].sum() > 0:
                try:
                    speaker, _, words = dataset.translate_item(tonumpy(speaker_padded[0:1, i]), None, words_padded_decode[i:i+1,:])
                    print('Real:', speaker[0], ' '.join(words[0]))
                    printed = 1
                except:
                    print('Exception Triggered. Received:', words_padded_decode[i:i+1,:])
                    break
            if printed == 0:
                break

            end_idx = np.where(greedy_responses[i,:]==eos)
            printed = 0
            if len(end_idx) > 0:
                end_idx = end_idx[0]
                if len(end_idx) > 0:
                    end_idx = end_idx[0]
                    if end_idx > 0:
                        speaker, _, words = dataset.translate_item(tonumpy(speaker_padded[0:1, i+1]), None, greedy_responses[i:i+1,:end_idx+1])
                        for _n in range(3):
                            ngram_counts[_n].update(nltk.ngrams(words[0], _n + 1))
                        print('Fake:', speaker[0], ' '.join(words[0]))
                        printed = 1
            if printed == 0:
                speaker, _, words = dataset.translate_item(tonumpy(speaker_padded[0:1, i+1]), None, greedy_responses[i:i+1,:])
                for _n in range(3):
                    ngram_counts[_n].update(nltk.ngrams(words[0], _n + 1))
                print('Fake:', speaker[0], ' '.join(words[0]))
            if words_padded_decode[i, 1].sum() == 0:
                break

    print('Dataset %s' % subdir)
    print('Average PPL: %f' % (sum_ppl / count_ppl))
    ngram_total = [sum(c.values()) for c in ngram_counts]
    for i in range(3):
        print('Most common %d-gram' % (i + 1))
        for k, v in ngram_counts[i].most_common(5):
            print(k, v / ngram_total[i])
