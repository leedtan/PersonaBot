
from torch.nn import Parameter
from functools import wraps

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
        wd_emb = cuda(wd_emb)
        usr_emb = cuda(usr_emb)
        length = cuda(length)

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

        return h[:, -2:].contiguous().view(batch_size, output_size), embed

class Context(NN.Module):
    def __init__(self,in_size, context_size, attention, num_layers=1,
                 attention_enabled = True):
        NN.Module.__init__(self)
        self._context_size = context_size
        self._in_size = in_size
        self._num_layers = num_layers
        self._attention_enabled = attention_enabled

        self.rnn = NN.LSTM(
                in_size,
                context_size,
                num_layers,
                bidirectional=False,
                )
        if self._attention_enabled:
            self.attention = attention
            self.attn_wd = NN.Sequential(
                NN.Linear(size_sentence * 2 + size_context, size_sentence),
                NN.LeakyReLU(),
                NN.Linear(size_sentence, 1))
            self.attn_sent = NN.Sequential(
                NN.Linear(size_sentence + size_sentence + context_size + size_usr * 2, 
                          size_sentence),
                NN.LeakyReLU(),
                NN.Linear(size_sentence, 1),)
            init_weights(self.attention)
            init_weights(self.attn_wd)
            init_weights(self.attn_sent)
            
        init_lstm(self.rnn)

    def zero_state(self, batch_size):
        lstm_h = tovar(T.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size))
        lstm_c = tovar(T.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size))
        initial_state = (lstm_h, lstm_c)
        return initial_state

    def forward(self, sent_encs, length, sentence_lengths_padded,
                wds_h, usrs_b, initial_state=None):
        if self._attention_enabled:
            attn = self.attention(sent_encs, length)
            # attn: batch_size, max_turns, sentence_encoding_size
        sent_encs = cuda(sent_encs)
        length = cuda(length)
        initial_state = cuda(initial_state)

        num_layers = self._num_layers
        batch_size = sent_encs.size()[0]
        context_size = self._context_size
        
        if initial_state is None:
            initial_state = self.zero_state(batch_size)
        sent_encs = sent_encs.permute(1,0,2)
        embed, (h, c) = dynamic_rnn(self.rnn, sent_encs, length, initial_state)
        embed = cuda(embed.permute(1,0,2).contiguous())
        # embed is now: batch_size, max_turns, context_size
        if self._attention_enabled:
            #ctx is batchsize, num_turns decode, size_context + size_sentence
            #wds_h maximum words in batch (real), batch_size * maximum turns in sample, size_sentence
            ctx = T.cat((embed, attn),2)
            batch_size, num_turns, _ = ctx.size()
            wds_in_sample, _, size_sentence = wds_h.size()
            wds_h_attn = wds_h.view(wds_in_sample, batch_size, num_turns, 
                                    size_sentence).permute(1,2,0,3).unsqueeze(1)
            #wds_h_attn is now batch, (TO BE USED FOR ATTN HEAD), 
            #num_turns_historical, words_in_sample, size_sentence
            wds_h_attn = wds_h.view(batch_size, 1, wds_in_sample * num_turns, size_sentence)
            wds_h_attn_expanded = wds_h_attn.expand(
                    batch_size, num_turns, wds_in_sample * num_turns, size_sentence)
            ctx_unsqueezed = ctx.view(batch_size, num_turns, size_context + size_sentence).unsqueeze(2)
            ctx_expanded = ctx_unsqueezed.expand(batch_size, num_turns, wds_in_sample * num_turns,
                             size_context + size_sentence)
            #ctx_expanded and wds_h_attn are
            #bs, num_turns, wds in sample*num_turns, size
            wds_h_attn_expanded = wds_h_attn_expanded.contiguous().view(-1, size_sentence)
            ctx_expanded = ctx_expanded.contiguous().view(-1, size_sentence + size_context)
            
            attn_wd = self.attn_wd(T.cat((ctx_expanded, wds_h_attn_expanded), 1).view(
                    -1, size_context + size_sentence*2)).view(
                    batch_size, num_turns, wds_in_sample, num_turns)
            #wipe out after sentence_lengths_padded in dimension 2 conditioned on batch and dim 3
            #wipe out after length in dimension 1 conditioned on batch
            #wipe out after index 1 in dimension 3 conditioned on length or length in dim 1
            size = attn_wd.size()
            mask = T.ones(*size)

            for i_b in range(size[0]):
                for i_head in range(size[1]):
                    for i_wd in range(size[2]):
                        for i_sent in range(size[3]):
                            if ((sentence_lengths_padded[i_b, i_sent] < i_wd)
                                    or (length[i_b] < i_head)
                                    or (i_head < i_sent)
                                    or (length[i_b] < i_sent)):
                                mask[i_b, i_head, i_wd, i_sent] = 0
            mask = tovar(mask)
            #mask2 = mask_3d(attn_wd[:,0,:,:].size(), sentence_lengths_padded)
            #mask13 = mask_4d(wds_b[:,0,:,:,:].size(), length , sentence_lengths_padded)
            
            wds_h_attn_expanded = wds_h_attn_expanded.view(
                    batch_size, num_turns, wds_in_sample, num_turns, size_sentence)
            attn_wd_masked = attn_wd * mask
            # FIXME: variable length softmax
            attn_wd_masked = weighted_softmax(
                    attn_wd_masked.permute(0, 1, 3, 2).contiguous().view(
                            batch_size * num_turns * num_turns, wds_in_sample), mask.permute(0,1,3,2).contiguous().view(
                            batch_size * num_turns * num_turns, wds_in_sample)).view(
                            batch_size, num_turns, num_turns,wds_in_sample,1).permute(0,1,3,2,4)
            at_weighted_wds = attn_wd_masked * wds_h_attn_expanded 
            at_weighted_sent = at_weighted_wds.sum(2)
            # (batch_size, num_turns, num_turns, sent_encoding_size)
            _, _, size_usr = usrs_b.size()
            
            usrs_b_expanded_for_messages = usrs_b.unsqueeze(1).expand(batch_size, num_turns, num_turns, size_usr)
            usrs_and_messages = T.cat((usrs_b_expanded_for_messages, at_weighted_sent),3)

            ctx_for_message_attn = ctx.unsqueeze(2).expand(
                    batch_size, num_turns, num_turns, size_context + size_sentence)
            if num_turns == 1:
                usrs_b_expanded_for_ctx = tovar(T.zeros(batch_size, 1, size_usr))
            else:
                usrs_b_expanded_for_ctx = T.cat([usrs_b[:, 1:, :], tovar(T.zeros(batch_size, 1, size_usr))], 1)
            usrs_b_expanded_for_ctx = usrs_b_expanded_for_ctx.contiguous().unsqueeze(2).expand(
                            batch_size, num_turns, num_turns, size_usr)
            usrs_and_ctx = T.cat((usrs_b_expanded_for_ctx, ctx_for_message_attn),3)

            # FIXME
            ctx_and_messages = T.cat((usrs_and_messages, usrs_and_ctx),3)
            #shape bs, turns, turns, ctx + sent * 2
            ctx_and_messages = ctx_and_messages.view(-1, size_context + size_sentence*2+size_usr*2)
            attn_rolled_up_sent = self.attn_sent(ctx_and_messages).view(
                    batch_size, num_turns, num_turns)
            #wipe out after length in dimension 1 conditioned on batch
            #wipe out after index 1 in dimension 2 conditioned on length
            size = attn_rolled_up_sent.size()
            mask_sent = T.ones(*size)

            for i_b in range(size[0]):
                for i_head in range(size[1]):
                    for i_sent in range(size[2]):
                        if ((length[i_b] < i_head)
                                or (i_head < i_sent)
                                or (length[i_b] < i_sent)):
                            mask_sent[i_b, i_head, i_sent] = 0
            mask_sent = tovar(mask_sent)
            attn_sent_masked = attn_rolled_up_sent * mask_sent
            # FIXME variable length softmax
            attn_sent_softmax = weighted_softmax(
                    attn_sent_masked.view(batch_size * num_turns, num_turns),
                    mask_sent.view(batch_size*num_turns, num_turns)).view(
                            batch_size, num_turns, num_turns, 1)
            usrs_and_sent_attnended = usrs_and_messages * attn_sent_softmax
            usrs_and_sent_rolled_up_to_ctx = usrs_and_sent_attnended.sum(2)
            ctx = T.cat((ctx, usrs_and_sent_rolled_up_to_ctx),2)
            
            #bs, num_turns (for attention head), num_turns, size_sentence
        else:
            ctx = embed[0]
        return ctx, h.contiguous()

def inverseHackTorch(tens):
    idx = [i for i in range(tens.size(2)-1,-1, -1)]
    idx = cuda(T.LongTensor(idx))
    inverted_tensor = tens[:,:,idx]
    return inverted_tensor
class Attention(NN.Module):
    def __init__(self, size_sentence, max_turns_allowed, num_layers = 1):
        NN.Module.__init__(self)
        self._size_sentence = size_sentence
        self._max_turns_allowed = max_turns_allowed
        self._num_layers = num_layers
        self.softmax = NN.Softmax()
        self.F = NN.Sequential(
            NN.Linear(size_sentence, size_sentence),
            NN.LeakyReLU(),
            NN.Linear(size_sentence, max_turns_allowed))
        init_weights(self.F)

    def forward(self, sent_encodings, turns):
        batch_size, num_turns, size_sentence = sent_encodings.size()
        sent_encodings = cuda(sent_encodings)
        max_turns_allowed = self._max_turns_allowed
        num_layers = self._num_layers
        attention_heads = self.F(sent_encodings.view(
                batch_size * num_turns, size_sentence))
        attention_heads = attention_heads.view(
                batch_size, -1, max_turns_allowed)[
                :,:,:num_turns].contiguous()
        #Attention heads is now:
        #batch_size by num_turns in sent by num_turns (max num turns)
        #will become
        #batch_size by num_turns by sent by 1
        #sent_encodings is:
        #batch_size by num_turns by size_sentence
        #sent_encodings will become:
        #batch_size by num_turns by (attention futures) by size_sentence
        #Need to chop off attention at
        #batch size by num_turns by (UP TO current sentence) by 1
        attention_heads = T.cat([
                T.cat((inverseHackTorch(attention_heads[:,i:i+1,:i+1]),
                       tovar(-1e8*T.ones((batch_size, 1, num_turns - i-1)))),2)
                if i < num_turns - 1 else inverseHackTorch(attention_heads[:,i:i+1,:i+1])
                for i in range(num_turns)], 1)
        attention_heads = attention_heads.view(
                batch_size * num_turns, -1)
        attention_heads = self.softmax(attention_heads)
        attention_heads = attention_heads.view(
                batch_size, num_turns, -1).unsqueeze(3)
        '''
        attn_shifted = T.cat([
                T.cat((inverseHackTorch(attention_heads[:,i,:i+1,:]).unsqueeze(1),
                T.zeros((batch_size, 1, size_sentence - i, 1))),2)
                if i < attention_heads.size()[1]-1 else 
                inverseHackTorch(attention_heads[:,i,:,:]).unsqueeze(1)
                for i in range(attention_heads.size()[1])],1)
        '''
        sent_encodings = sent_encodings.unsqueeze(2)
        return (sent_encodings * attention_heads).sum(1)

class Decoder(NN.Module):
    def __init__(self,size_usr, size_wd, context_size, size_sentence, num_words, max_len_generated ,beam_size,
                 state_size = None, num_layers=1, attention_enabled=True):
        NN.Module.__init__(self)
        self._num_words = num_words
        self._beam_size = beam_size
        self._max_len_generated = max_len_generated
        self._context_size = context_size
        self._size_wd = size_wd
        self._size_usr = size_usr
        self._num_layers = num_layers
        self._attention_enabled = attention_enabled
        if self._attention_enabled:
            in_size = size_usr + size_wd + context_size + size_sentence*2 + size_usr
        else:
            in_size = size_usr + size_wd + context_size
        if state_size == None:
            state_size = in_size
        self._state_size = state_size

        self.rnn = NN.LSTM(
                in_size,
                state_size,
                num_layers,
                bidirectional=False,
                )
        self.softmax = HierarchicalLogSoftmax(state_size, np.int(np.sqrt(num_words)), num_words)
        init_lstm(self.rnn)

    def zero_state(self, batch_size):
        lstm_h = tovar(T.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size))
        lstm_c = tovar(T.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size))
        initial_state = (lstm_h, lstm_c)
        return initial_state

    def forward(self, context_encodings, wd_emb, usr_emb, length, wd_target=None, initial_state=None):
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
        length = cuda(length)
        wd_target = cuda(wd_target)
        initial_state = cuda(initial_state)

        num_layers = self._num_layers
        batch_size = wd_emb.size()[0]
        maxlenbatch = wd_emb.size()[1]
        maxwordsmessage = wd_emb.size()[2]
        context_size = self._context_size
        size_wd = self._size_wd
        size_usr = self._size_usr
        state_size = self._state_size
        if initial_state is None:
            initial_state = self.zero_state(batch_size * maxlenbatch)
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
        maxwordsmessage = embed.size()[0]
        embed = embed.permute(1, 0, 2).contiguous().view(batch_size, maxlenbatch, maxwordsmessage, state_size)

        if wd_target is None:
            out = self.softmax(embed.view(-1, state_size))
            out = out.view(batch_size, maxlenbatch, -1, self._num_words)
            log_prob = None
        else:
            target = T.cat((wd_target[:, :, 1:], tovar(T.zeros(batch_size, maxlenbatch, 1)).long()), 2)
            out = self.softmax(embed.view(-1, state_size), target.view(-1))
            out = out.view(batch_size, maxlenbatch, maxwordsmessage)
            mask = (target != 0).float()
            out = out * mask
            log_prob = out.sum() / mask.sum()

        if log_prob is None:
            return out, (h, c)#.contiguous().view(batch_size, maxlenbatch, maxwordsmessage, -1)
        else:
            return out, log_prob, (h, c)

    def get_n_best_next_words(self, embed_seq, current_state, n=1):
        """

        :param embed_seq: batch_size x (usr_emb_size + w_emb_size + context_emb_size)
        :param current_state: num_layers * num_directions x batch_size x hidden_size
        :param n: int, return top n words.
        :return: batch_size x n
        """
        embed_seq = cuda(embed_seq)
        current_state = cuda(current_state)

        batch_size = embed_seq.size(0)
        embed, current_state = self.rnn(embed_seq.unsqueeze(0), current_state)
        embed = embed.permute(1, 0, 2).contiguous().view(batch_size, self._state_size)
        out = self.softmax(embed.squeeze())
        val, indexes = out.topk(n, 1)
        return val, indexes, current_state

    def mat_idx_vector_to_vector(self, mat, vec):
        """

        :param mat: a matrix of size l lines x m cols
        :param vec: vector of size l
        :return: vector made of M[i, l[i]] i in [1,l]
        """
        out = T.LongTensor(mat.size(0))
        for i in range(mat.size(0)):
            out[i] = mat[i,vec[i]]

        return out

    def greedyGenerate(self, context_encodings, usr_emb, word_emb, dataset):
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
        lstm_h = tovar(T.zeros(num_layers, batch_size, state_size))
        lstm_c = tovar(T.zeros(num_layers, batch_size, state_size))
        current_state = (lstm_h, lstm_c)

        # Initial word of response : Start token
        init_word = tovar(T.LongTensor(batch_size).fill_(dataset.index_word(START)))
        # End of generated sentence : EOS token
        stop_word = cuda(T.LongTensor(batch_size).fill_(dataset.index_word(EOS)))

        current_w = init_word
        output = current_w.data.unsqueeze(1)

        while not stop_word.equal(current_w.data.squeeze()) and output.size(1) < max_len_generated:
            current_w_emb = word_emb(current_w.squeeze())
            embed_seq = T.cat((usr_emb, current_w_emb, context_encodings), 1)
            _, current_w, current_state= self.get_n_best_next_words(embed_seq, current_state)
            output = T.cat((output, current_w.data), 1)

        output = cuda(output)
        return output

    def viterbiGenerate(self, context_encodings, usr_emb, word_emb, dataset):
        """
        :param context_encodings: (batch_size x context_size)
        :param usr_emb: (batch_size x usr_emb_size)
        :param word_emb:  idx to vector word embedder.
        :param dataset:  dataset to get EOS/START tokens ids
        :param beam_size:  width of beam search.
        :return: response : (batch_size x max_response_length)
        """
        context_encodings = cuda(context_encodings)
        usr_emb = cuda(usr_emb)
        word_emb = cuda(word_emb)

        beam_size = self._beam_size
        max_len_generated = self._max_len_generated
        num_layers = self._num_layers
        state_size = self._state_size
        batch_size = context_encodings.size(0)

        # End of generated sentence : EOS token
        initial_word = tovar(T.LongTensor(batch_size).fill_(dataset.index_word(START)))
        stop_word = tovar(T.LongTensor(batch_size).fill_(dataset.index_word(EOS)))

        usr_emb = usr_emb.unsqueeze(1)
        context_encodings = context_encodings.unsqueeze(1)

        # Viterbi tensors :
        # dim 0 : 0 -> current word index, 1 -> previous word index leading to this word.
        s_idx_w_idx = tovar(T.LongTensor(2, 1, batch_size, beam_size).fill_(dataset.index_word(START)))
        s_idx_w_idx_logproba= tovar(T.FloatTensor(1, batch_size, beam_size).fill_(0))
        s_idx_w_idx_lstm_h = tovar(T.zeros(num_layers, batch_size, beam_size, state_size))
        s_idx_w_idx_lstm_c = tovar(T.zeros(num_layers, batch_size, beam_size, state_size))

        # First step from START to first probable words :

        lstm_h = tovar(T.zeros(num_layers, batch_size, state_size))
        lstm_c = tovar(T.zeros(num_layers, batch_size, state_size))

        current_w_emb = word_emb(initial_word.unsqueeze(1))
        embed_seq = T.cat((usr_emb, current_w_emb, context_encodings), 2)
        transition_probabilities, transition_index, current_state = self.get_n_best_next_words(
                embed_seq.squeeze(), (lstm_h, lstm_c), beam_size)
        next_wrd = transition_index.unsqueeze(0)
        prev_idx =  tovar(T.zeros(transition_index.size()).long().unsqueeze(0))
        nxt_s_idx_w_idx = T.cat((next_wrd, prev_idx), 0)
        nxt_s_idx_w_idx = nxt_s_idx_w_idx.unsqueeze(1)
        s_idx_w_idx= T.cat((s_idx_w_idx, nxt_s_idx_w_idx), 1)
        s_idx_w_idx_logproba = T.cat((s_idx_w_idx_logproba, transition_probabilities.unsqueeze(0)), 0)
        s_idx_w_idx_lstm_h = current_state[0].unsqueeze(2).expand_as(s_idx_w_idx_lstm_h).contiguous()
        s_idx_w_idx_lstm_c = current_state[1].unsqueeze(2).expand_as(s_idx_w_idx_lstm_h).contiguous()

        usr_emb = usr_emb.expand(usr_emb.size(0), beam_size, usr_emb.size(2))
        context_encodings = context_encodings.expand(context_encodings.size(0), beam_size, context_encodings.size(2))

        s_idx = 1

        # Add constraint argmax probability is end word for all batch.

        while s_idx < max_len_generated - 1:
            current_w_emb = word_emb(s_idx_w_idx[0,s_idx])
            embed_seq = T.cat((usr_emb, current_w_emb, context_encodings), 2)
            transition_probabilities, transition_index, current_state = self.get_n_best_next_words(
                    embed_seq.view(-1, embed_seq.size(2)), (s_idx_w_idx_lstm_h.view(num_layers, -1, state_size), 
                                   s_idx_w_idx_lstm_c.view(num_layers, -1, state_size)), beam_size)

            transition_index = transition_index.view(batch_size, -1)

            transition_probabilities = transition_probabilities.view(batch_size, beam_size, beam_size)
            transition_probabilities.add_(s_idx_w_idx_logproba[s_idx].unsqueeze(2).expand_as(transition_probabilities))
            transition_probabilities = transition_probabilities.view(batch_size, -1)

            lstm_h = current_state[0].view(num_layers, -1, beam_size, state_size)
            lstm_c = current_state[1].view(num_layers, -1, beam_size, state_size)

            best_transition_probabilities, best_transition_indexes = transition_probabilities.topk(beam_size, 1)
            next_words = T.LongTensor(best_transition_indexes.size())

            best_transition_indexes = best_transition_indexes.data
            best_transition_return_index = best_transition_indexes / 13

            for i in range(best_transition_indexes.size(0)):
                next_words[i] = transition_index[i][best_transition_indexes[i]].data
                s_idx_w_idx_lstm_h[:,i,:,:].data = lstm_h[:,:,best_transition_return_index[i],:][:,i,:,:].data
                s_idx_w_idx_lstm_c[:, i, :, :].data = lstm_c[:, :, best_transition_return_index[i], :][:, i, :, :].data

            s_idx_w_idx_logproba = T.cat((s_idx_w_idx_logproba, best_transition_probabilities.unsqueeze(0)), 0)

            best_transition_indexes = best_transition_indexes.unsqueeze(0).unsqueeze(0)
            next_words = next_words.unsqueeze(0).unsqueeze(0)
            s_idx_w_idx = T.cat((s_idx_w_idx, T.cat((next_words, best_transition_indexes / beam_size), 0)), 1)

            s_idx += 1

        # Now we go through the viterbi matrix backward to get the best sentence :

        s_idx_w_idx = s_idx_w_idx.data
        answers = T.zeros(batch_size, s_idx + 1).long()

        best_words, best_idx = s_idx_w_idx[0, s_idx].max(1)
        idx_previous = self.mat_idx_vector_to_vector(s_idx_w_idx[1, s_idx], best_idx)
        answers[:, s_idx] = best_words

        while s_idx >= 1:
            s_idx -= 1
            answers[:, s_idx] = self.mat_idx_vector_to_vector(s_idx_w_idx[0,s_idx], idx_previous)
            idx_previous =  self.mat_idx_vector_to_vector(s_idx_w_idx[1, s_idx], idx_previous)

        answers = cuda(answers)
        return answers

parser = argparse.ArgumentParser(description='Ubuntu Dialogue dataset parser')
parser.add_argument('--dataroot', type=str,default='ubuntu', help='Root of the data downloaded from github')
parser.add_argument('--metaroot', type=str, default='.', help='Root of meta data')
parser.add_argument('--vocabsize', type=int, default=159996, help='Vocabulary size')
parser.add_argument('--gloveroot', type=str,default='data', help='Root of the data downloaded from github')
parser.add_argument('--outputdir', type=str, default ='outputs',help='output directory')
parser.add_argument('--logdir', type=str, default='logs', help='log directory')
parser.add_argument('--encoder_layers', type=int, default=2)
parser.add_argument('--decoder_layers', type=int, default=2)
parser.add_argument('--context_layers', type=int, default=2)
parser.add_argument('--size_context', type=int, default=128)
parser.add_argument('--size_sentence', type=int, default=128)
parser.add_argument('--decoder_size_sentence', type=int, default=256)
parser.add_argument('--decoder_beam_size', type=int, default=4)
parser.add_argument('--decoder_max_generated', type=int, default=60)
parser.add_argument('--size_usr', type=int, default=32)
parser.add_argument('--size_wd', type=int, default=50)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--gradclip', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--modelname', type=str, default = '')
parser.add_argument('--modelnamesave', type=str, default='')
parser.add_argument('--modelnameload', type=str, default='')
parser.add_argument('--loaditerations', type=int, default=0)
parser.add_argument('--max_sentence_length_allowed', type=int, default=60)
parser.add_argument('--max_turns_allowed', type=int, default=5)
parser.add_argument('--num_loader_workers', type=int, default=4)
parser.add_argument('--adversarial_sample', type=int, default=1)
parser.add_argument('--attention_enabled', type=bool, default=True)
parser.add_argument('--emb_gpu_id', type=int, default=0)
parser.add_argument('--ctx_gpu_id', type=int, default=0)
parser.add_argument('--enc_gpu_id', type=int, default=0)
parser.add_argument('--dec_gpu_id', type=int, default=0)
args = parser.parse_args()

dataset = UbuntuDialogDataset(args.dataroot,
                              wordcount_pkl=args.metaroot + '/wordcount.pkl',
                              usercount_pkl=args.metaroot + '/usercount.pkl',
                              turncount_pkl=args.metaroot + '/turncount.pkl',
                              max_sentence_lengths_pkl=args.metaroot + '/max_sentence_lengths.pkl',
                              max_sentence_length_allowed=args.max_sentence_length_allowed,
                              max_turns_allowed=args.max_turns_allowed,
                              vocab_size=args.vocabsize)
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
log_train = logdirs(args.logdir, modelnamesave)

train_writer = TF.summary.FileWriter(log_train)

vcb = dataset.vocab
usrs = dataset.users
num_usrs = len(usrs)
vcb_len = len(vcb)
num_words = vcb_len
size_usr = args.size_usr
size_wd = args.size_wd
size_sentence = args.size_sentence
size_context = args.size_context
decoder_size_sentence = args.decoder_size_sentence
decoder_beam_size = args.decoder_beam_size
decoder_max_generated = args.decoder_max_generated

user_emb = cuda(NN.Embedding(num_usrs+1, size_usr, padding_idx = 0))
word_emb = cuda(NN.Embedding(vcb_len+1, size_wd, padding_idx = 0))
#word_emb.weight.data.copy_(init_glove(word_emb, vcb, dataset._ivocab, args.gloveroot))
enc = cuda(Encoder(size_usr, size_wd, size_sentence, num_layers = args.encoder_layers))
attention = cuda(Attention(size_sentence, args.max_turns_allowed, num_layers = 1))
context = cuda(Context(size_sentence, size_context, attention, 
               num_layers = args.context_layers, 
               attention_enabled = args.attention_enabled))
decoder = cuda(Decoder(size_usr, size_wd, size_context, size_sentence, num_words+1,
               decoder_max_generated, decoder_beam_size, 
               state_size=decoder_size_sentence, 
               num_layers = args.decoder_layers,
               attention_enabled = args.attention_enabled))
params = sum([list(m.parameters()) for m in [user_emb, word_emb, enc, context, decoder]], [])
opt = T.optim.Adam(params, lr=args.lr)


dataloader = UbuntuDialogDataLoader(dataset, args.batchsize, num_workers=args.num_loader_workers)

itr = args.loaditerations
epoch = 0
usr_std = wd_std = sent_std = ctx_std = 1e-7
adv_emb_diffs = []
adv_sent_diffs = []
adv_ctx_diffs = []
adv_emb_scales = []
adv_sent_scales = []
adv_ctx_scales = []
if modelnameload:
    if len(modelnameload) > 0:
        user_emb = T.load('%s-user_emb-%08d' % (modelnameload, args.loaditerations))
        word_emb = T.load('%s-word_emb-%08d' % (modelnameload, args.loaditerations))
        enc = T.load('%s-enc-%08d' % (modelnameload, args.loaditerations))
        context = T.load('%s-context-%08d' % (modelnameload, args.loaditerations))
        decoder = T.load('%s-decoder-%08d' % (modelnameload, args.loaditerations))
adv_style = 0
scatter_entropy_freq = 200
while True:
    epoch += 1
    for item in dataloader:
        if itr % scatter_entropy_freq == 0:
            adv_style = 1 - adv_style
            adjust_learning_rate(opt, args.lr / np.sqrt(1 + itr / 10000))
        itr += 1
        turns, sentence_lengths_padded, speaker_padded, \
            addressee_padded, words_padded, words_reverse_padded = item
        words_padded = tovar(words_padded)
        words_reverse_padded = tovar(words_reverse_padded)
        speaker_padded = tovar(speaker_padded)
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

        if itr % 10 == 1 and args.adversarial_sample == 1:
            scale = float(np.exp(-np.random.uniform(2, 6)))
            wds_adv, usrs_adv, loss_adv = adversarial_word_users(wds_b, usrs_b, turns,
               size_wd,batch_size,size_usr,
               sentence_lengths_padded, enc, 
               context,words_padded, decoder, usr_std, wd_std, scale=scale, style=adv_style)
            wds_b = tovar((wds_b + tovar(wds_adv)).data)
            usrs_b = tovar((usrs_b + tovar(usrs_adv)).data)
        max_turns = turns.max()
        max_words = wds_b.size()[2]
        encodings, wds_h = enc(wds_b.view(batch_size * max_turns, max_words, size_wd),
                usrs_b.view(batch_size * max_turns, size_usr), 
                sentence_lengths_padded.view(-1))
        if itr % 10 == 4 and args.adversarial_sample == 1:
            scale = float(np.exp(-np.random.uniform(2, 6)))
            wds_adv, usrs_adv, enc_adv, loss_adv = adversarial_encodings_wds_usrs(encodings, batch_size, 
                    wds_b,usrs_b,max_turns, context, turns, 
                    sentence_lengths_padded, words_padded, decoder,
                    usr_std, wd_std, sent_std, wds_h, scale=scale, style=adv_style)
            wds_b = tovar((wds_b + tovar(wds_adv)).data)
            usrs_b = tovar((usrs_b + tovar(usrs_adv)).data)
            encodings = tovar((encodings + tovar(enc_adv).view_as(encodings)).data)
        encodings = encodings.view(batch_size, max_turns, -1)
        #attn = attention(encodings)
        ctx, _ = context(encodings, turns, sentence_lengths_padded, wds_h.contiguous(), usrs_b)
        if itr % 10 == 7 and args.adversarial_sample == 1:
            scale = float(np.exp(-np.random.uniform(2, 6)))
            wds_adv, usrs_adv, ctx_adv, loss_adv = adversarial_context_wds_usrs(ctx, sentence_lengths_padded,
                      wds_b,usrs_b,words_padded, decoder,
                      usr_std, wd_std, ctx_std, wds_h, scale=scale, style=adv_style)
            wds_b = tovar((wds_b + tovar(wds_adv)).data)
            usrs_b = tovar((usrs_b + tovar(usrs_adv)).data)
            ctx = tovar((ctx + tovar(ctx_adv)).data)
        max_output_words = sentence_lengths_padded[:, 1:].max()
        wds_b_decode = wds_b[:,1:,:max_output_words].contiguous()
        usrs_b_decode = usrs_b[:,1:]
        words_flat = words_padded[:,1:,:max_output_words].contiguous()
        # Training:
        prob, log_prob, _ = decoder(ctx[:,:-1:], wds_b_decode,
                                 usrs_b_decode, sentence_lengths_padded[:,1:], words_flat)
        loss = -log_prob
        opt.zero_grad()
        loss.backward()
        grad_norm = clip_grad(params, args.gradclip)
        loss, grad_norm = tonumpy(loss, grad_norm)
        loss = loss[0]
        opt.step()
        if itr % 10 == 1 and args.adversarial_sample == 1:
            adv_emb_diffs.append(loss_adv - loss)
            adv_emb_scales.append(scale)
            train_writer.add_summary(
                TF.Summary(value=[TF.Summary.Value(tag='wd_usr_adv_diff', simple_value=loss_adv - loss)]),itr)
        if itr % 10 == 4 and args.adversarial_sample == 1:
            adv_sent_diffs.append(loss_adv - loss)
            adv_sent_scales.append(scale)
            train_writer.add_summary(
                TF.Summary(value=[TF.Summary.Value(tag='enc_adv_diff', simple_value=loss_adv - loss)]),itr)
        if itr % 10 == 7 and args.adversarial_sample == 1:
            adv_ctx_diffs.append(loss_adv - loss)
            adv_ctx_scales.append(scale)
            train_writer.add_summary(
                TF.Summary(value=[TF.Summary.Value(tag='ctx_adv_diff', simple_value=loss_adv - loss)]),itr)
        mask = mask_4d(wds_b.size(), turns , sentence_lengths_padded)
        wds_dist = wds_b* mask
        mask = mask_3d(usrs_b.size(), turns)
        usrs_dist = usrs_b * mask
        mask = mask_3d(encodings.size(), turns)
        sent_dist = encodings * mask
        mask = mask_3d(ctx.size(), turns)
        ctx_dist = ctx * mask
        wds_dist, usrs_dist, sent_dist, ctx_dist = tonumpy(wds_dist, usrs_dist, sent_dist, ctx_dist)
        if itr % 10 == 9:
            wd_std = float(np.nanstd(wds_dist))
            usr_std = float(np.nanstd(usrs_dist))
            sent_std = float(np.nanstd(sent_dist))
            ctx_std = float(np.nanstd(ctx_dist))
            train_writer.add_summary(
                    TF.Summary(
                        value=[
                            TF.Summary.Value(tag='loss', simple_value=loss),
                            TF.Summary.Value(tag='grad_norm', simple_value=grad_norm),
                            TF.Summary.Value(tag='wd_std', simple_value=wd_std),
                            TF.Summary.Value(tag='usr_std', simple_value=usr_std),
                            TF.Summary.Value(tag='sent_std', simple_value=sent_std),
                            TF.Summary.Value(tag='ctx_std', simple_value=ctx_std),
                            ]
                        ),
                    itr
                    )
        # Beam search test
        #FIXME PLEASE. BREAKS ON ATTENTION
        '''
        if itr % 100 == 0:
            words = tonumpy(words_padded.data[0, ::2])
            sentence_lengths = tonumpy(sentence_lengths_padded[0, ::2])
            initiator = speaker_padded.data[0, 0]
            respondent = speaker_padded.data[0, 1]
            words_nopad = [list(words[i, :sentence_lengths[i]]) for i in range(turns[0] // 2)]
            dialogue, scores = test(dataset, enc, context, decoder, word_emb, user_emb, words_nopad,
                                    initiator, respondent, args.max_sentence_length_allowed)
            _, _, dialogue_strings = dataset.translate_item(None, None, dialogue)
            for i, (d, ds, s) in enumerate(zip(dialogue, dialogue_strings, scores)):
                print('REAL' if i % 2 == 0 else 'FAKE', ds, d, s)
        '''
        
        if itr % scatter_entropy_freq == 0:
            prob, _ = decoder(ctx[:1,:-1], wds_b_decode[:1,:,:],
                                 usrs_b_decode[:1:], sentence_lengths_padded[:1,1:])
            #Entropy defined as H here:https://en.wikipedia.org/wiki/Entropy_(information_theory)
            mask = mask_4d(prob.size(), turns[:1] -1 , sentence_lengths_padded[:1,1:])
            Entropy = (prob.exp() * prob * -1) * mask
            Entropy_per_word = Entropy.sum(-1)
            Entropy_per_word = tonumpy(Entropy_per_word)[0]
            #E_mean = tonumpy(Entropy_per_word.sum() / mask.sum())[0]
            #E_mean, E_std, E_max, E_min = tonumpy(
            #        Entropy_per_word.mean(), Entropy.std(), Entropy.max(), Entropy.min())
            E_mean = np.nanmean(Entropy_per_word)
            train_writer.add_summary(
                TF.Summary(
                    value=[
                        TF.Summary.Value(tag='Entropy_mean', simple_value=E_mean)
                        ]
                    ),
                itr
            )
            '''
            E_mean, E_std, E_max, E_min = \
                    np.nanmean(Entropy_per_word), np.nanstd(Entropy_per_word), \
                    np.nanmax(Entropy_per_word), np.nanmin(Entropy_per_word)
            
            
            #E_mean, E_std, E_max, E_min = [e[0] for e in [E_mean, E_std, E_max, E_min]]
            train_writer.add_summary(
                TF.Summary(
                    value=[
                        TF.Summary.Value(tag='Entropy_mean', simple_value=E_mean),
                        TF.Summary.Value(tag='Entropy_std', simple_value=E_std),
                        TF.Summary.Value(tag='Entropy_max', simple_value=E_max),
                        TF.Summary.Value(tag='Entropy_min', simple_value=E_min),
                        ]
                    ),
                itr
                )
            '''
            if args.adversarial_sample == 1:
                add_scatterplot(train_writer, losses=[adv_emb_diffs, adv_sent_diffs, adv_ctx_diffs], 
                                scales=[adv_emb_scales, adv_sent_scales, adv_ctx_scales], 
                                names=['embeddings', 'sentence', 'context'], itr = itr, 
                                log_dir = args.logdir, tag = 'scatterplot', style=adv_style)
                adv_emb_diffs = []
                adv_sent_diffs = []
                adv_ctx_diffs = []
                adv_emb_scales = []
                adv_sent_scales = []
                adv_ctx_scales = []        
        if itr % 100 == 0:
            greedy_responses = decoder.greedyGenerate(ctx[:1,:,:].view(-1, size_context + size_sentence * 2 + size_usr),
                                                      usrs_b[:1,:,:].view(-1, size_usr), 
                                                      word_emb, dataset)
            #print('REAL:',dataset.translate_item(None, None, tonumpy(words_padded[:1,:,:])))
            greedy_responses = tonumpy(greedy_responses)
            eos = dataset.index_word(EOS)
            words_padded_decode = tonumpy(words_padded[0,:,:])
            for i in range(greedy_responses.shape[0]):
                end_idx = np.where(greedy_responses[i,:]==eos)
                printed = 0
                if len(end_idx) > 0:
                    end_idx = end_idx[0]
                    if len(end_idx) > 0:
                        end_idx = end_idx[0]
                        if end_idx > 0:
                            print('Fake:',dataset.translate_item(None, None, greedy_responses[i:i+1,:end_idx+1]))
                            printed = 1
                if printed == 0:
                    print('Fake:',dataset.translate_item(None, None, greedy_responses[i:i+1,:]))
                    
                end_idx = np.where(words_padded_decode[i,:]==eos)
                printed = 0
                if len(end_idx) > 0:
                    end_idx = end_idx[0]
                    if len(end_idx) > 0:
                        end_idx = end_idx[0]
                        if end_idx > 0:
                            print('Real:',dataset.translate_item(None, None, words_padded_decode[i:i+1,:end_idx+1]))
                            printed = 1
                if printed == 0:
                    try:
                        print('Real:',dataset.translate_item(None, None, words_padded_decode[i:i+1,:]))
                    except:
                        print('Exception Triggered. Received:', words_padded_decode[i:i+1,:])
        
        if itr % 10000 == 0:
            T.save(user_emb, '%s-user_emb-%08d' % (modelnamesave, itr))
            T.save(word_emb, '%s-word_emb-%08d' % (modelnamesave, itr))
            T.save(enc, '%s-enc-%08d' % (modelnamesave, itr))
            T.save(context, '%s-context-%08d' % (modelnamesave, itr))
            T.save(decoder, '%s-decoder-%08d' % (modelnamesave, itr))
        if itr % 10 == 0:
            print('Epoch', epoch, 'Iteration', itr, 'Loss', tonumpy(loss), 'PPL', 2 ** tonumpy(loss))

    
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
