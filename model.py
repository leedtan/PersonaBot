
from torch.nn import Parameter
from functools import wraps
from nltk.translate import bleu_score

import time
import torch as T
import torch.nn as NN
import torch.nn.functional as F
import torch.nn.init as INIT
import tensorflow as tf
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tensorflow import nn
import numpy.random as RNG
from numpy import rate
from tensorflow.contrib.rnn import GRUCell
from tensorflow import layers
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
tf.reset_default_graph()

def MLP(x, hiddens, output_size, name, reuse = False):
    prev_layer = x
    for idx, l in enumerate(hiddens):
        prev_layer = lrelu(layers.dense(prev_layer, l, name=name+'_' +str(idx), reuse = reuse))
    output = layers.dense(prev_layer, output_size, name = name + 'final', reuse = reuse)
    return output

def print_greedy_decode(words_padded, greedy_responses, greedy_values):
    words_padded_decode = words_padded[0,:,:]
    greedy_responses = np.stack(greedy_responses,1)
    greedy_values = np.stack(greedy_values,1)
    for i in range(greedy_responses.shape[0]-1):
        end_idx = np.where(greedy_responses[i,:] == eos)
        printed = 0
        if len(end_idx) > 0:
            end_idx = end_idx[0]
            if len(end_idx) > 0:
                end_idx = end_idx[0]
                if end_idx > 0:
                    speaker, _, words = dataset.translate_item(
                            speaker_padded[0:1, i], None, words_padded_decode[i:i+1,:end_idx+1])
                    print('Real:', speaker[0], ' '.join(words[0]))
                    printed = 1
            if printed == 0 and words_padded_decode[i, 1].sum() > 0:
                try:
                    speaker, _, words = dataset.translate_item(
                            speaker_padded[0:1, i], None, words_padded_decode[i:i+1,:])
                    print('Real:', speaker[0], ' '.join(words[0]))
                    printed = 1
                except:
                    print('Exception Triggered. Received:', words_padded_decode[i:i+1,:])
                    break
            if printed == 0:
                break

<<<<<<< HEAD
            end_idx = np.where(greedy_responses[i,:]==eos)
            printed = 0
            if len(end_idx) > 0:
                end_idx = end_idx[0]
                if len(end_idx) > 0:
                    end_idx = end_idx[0]
                    if end_idx > 0:
                        speaker, _, words = dataset.translate_item(
                                speaker_padded[0:1, i+1], None, greedy_responses[i:i+1,:end_idx+1])
                        print('Fake:', speaker[0], ' '.join(words[0]))
                        printed = 1
            if printed == 0:
                try:
                    speaker, _, words = dataset.translate_item(
                            speaker_padded[0:1, i+1], None, greedy_responses[i:i+1,:])
                    print('Fake:', speaker[0], ' '.join(words[0]))
                except:
                    #If model has a bug, nans show up, somehow this crashes.
                    continue
            if words_padded_decode[i, 1].sum() == 0:
                break
    
def lrelu(x):
    return tf.maximum(x, .1*x)

class Attention():
    def __init__(self,head, history, head_shape, history_shape, head_expanded_shape, history_expanded_shape,
                 head_expand_dims, history_expand_dims, history_rollup_dims, name=''):
        head_size, history_size = head_shape[-1], history_shape[-1]
        head_expanded, history_expanded = head, history
        self.W = tf.Variable(1e-5*tf.random_normal([head_size, history_size]), name = name)
        for dim, value in head_expand_dims:
            head_expanded = tf.expand_dims(head_expanded, dim)
            tiles = [1 for _ in head_expanded.get_shape()]
            tiles[dim] = value
            head_expanded = tf.tile(head_expanded, tiles)
            
        for dim, value in history_expand_dims:
            history_expanded = tf.expand_dims(history_expanded, dim)
            tiles = [1 for _ in history_expanded.get_shape()]
            tiles[dim] = value
            history_expanded = tf.tile(history_expanded, tiles)
        
        #head_expanded_shape = head_expanded.get_shape()
        #history_expanded_shape = history_expanded.get_shape()
        
        head_flat = tf.reshape(head_expanded, [-1, head_size])
        history_flat = tf.reshape(history_expanded, [-1, history_size])
        headtimesw = tf.matmul(head_flat, self.W)
        headwhistory = headtimesw * history_flat
        headwhistorysqrt = tf.sqrt(tf.abs(headwhistory)) * tf.sign(headwhistory)
        self.attn_weights = tf.exp(tf.tanh(tf.reduce_sum(headwhistory/1000, -1))*10-5)
        self.attn_weights_shaped = tf.reshape(tf.tile(tf.expand_dims(
                self.attn_weights, -1), [1, history_size]), history_expanded_shape)
        #tile_dims = [1 for _  in history_expanded_shape]
        #tile_dims[history_rollup_dim] = history_expanded_shape[history_rollup_dim]
        self.attn_weights_reduced = self.attn_weights_shaped
        for dim, _ in history_rollup_dims[::-1]:
            self.attn_weights_reduced = tf.reduce_sum(self.attn_weights_reduced/100, dim)
        
        self.attn_weights_normalization = self.attn_weights_reduced
        for dim, val in history_rollup_dims:
            self.attn_weights_normalization = tf.expand_dims(self.attn_weights_normalization, dim)
            tiles = [1 for _ in self.attn_weights_normalization.get_shape()]
            tiles[dim] = val
            self.attn_weights_normalization = tf.tile(self.attn_weights_normalization, tiles)
=======
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
        self.context_attn_plot_path = "plot/context_attn_%d.png"

        self.rnn = NN.LSTM(
                in_size,
                context_size,
                num_layers,
                bidirectional=False,
                )
        if self._attention_enabled:
            self.attention = attention
            self.attn_wd = NN.Sequential(
                NN.Linear(size_sentence + size_context * 2, size_context),
                NN.LeakyReLU(),
                NN.Linear(size_context, size_context),
                NN.LeakyReLU(),
                NN.Linear(size_context, 1))
            self.attn_sent = NN.Sequential(
                NN.Linear(size_sentence + size_context + context_size + size_usr * 2, 
                          size_context),
                NN.LeakyReLU(),
                NN.Linear(size_context, size_context),
                NN.LeakyReLU(),
                NN.Linear(size_context, 1),)
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
                wds_h, usrs_b, plot=False, sentence_string=None, initial_state=None):
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
            attn, attn_weight = self.attention(embed, length)
            if plot:
                plot_attention(attn_weight=attn_weight, attended_over=sentence_string, print_path=self.context_attn_plot_path)
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
            ctx_unsqueezed = ctx.view(batch_size, num_turns, size_context*2).unsqueeze(2)
            ctx_expanded = ctx_unsqueezed.expand(batch_size, num_turns, wds_in_sample * num_turns,
                             size_context*2)
            #ctx_expanded and wds_h_attn are
            #bs, num_turns, wds in sample*num_turns, size
            wds_h_attn_expanded = wds_h_attn_expanded.contiguous().view(-1, size_sentence)
            ctx_expanded = ctx_expanded.contiguous().view(-1, size_context*2)
>>>>>>> ab3bd387d51524c6d786a6a1bc6f0a837c68e09c
            
        self.attn_weights_normalized = self.attn_weights_shaped / self.attn_weights_normalization
        self.attended_history =  self.attn_weights_normalized * history_expanded
        history_rollup_dimensions = [d for d, v in history_rollup_dims]
        self.attended_history_rolledup = lrelu(tf.reduce_sum(self.attended_history, history_rollup_dimensions))
        
    
        
    
    

def encode_sentence(x, num_layers, size, lengths, cells, initial_states):
    prev_layer = x
    for idx in range(num_layers):
        h,c = nn.bidirectional_dynamic_rnn(
            cell_fw=cells[idx][0], cell_bw=cells[idx][1],
            inputs=prev_layer,
            sequence_length=lengths,
            initial_state_fw=initial_states[idx][0],
            initial_state_bw=initial_states[idx][1],
            dtype=tf.float32,
            parallel_iterations=None,
            swap_memory=False,
            time_major=False,
            scope='enc' + str(idx)
            )
        prev_layer = lrelu(tf.concat(h, -1))
    output_layer = prev_layer[:,-1,:]
    return output_layer, prev_layer

def encode_context(x, num_layers, size, lengths, cells,initial_states, name='ctx'):
    prev_layer = x
    for idx in range(num_layers):
        prev_layer,c = nn.dynamic_rnn(
            cell=cells[idx],
            inputs=prev_layer,
            sequence_length=lengths,
            initial_state=initial_states[idx],
            dtype=tf.float32,
            parallel_iterations=None,
            swap_memory=False,
            time_major=False,
            scope=name + str(idx)
            )
        prev_layer = lrelu(prev_layer)
    return prev_layer

def decode(x, num_layers, size, lengths, cells,initial_states, name='dec'):
    prev_layer = x
    for idx in range(num_layers):
        prev_layer,c = nn.dynamic_rnn(
            cell=cells[idx],
            inputs=prev_layer,
            sequence_length=lengths,
            initial_state=initial_states[idx],
            dtype=tf.float32,
            parallel_iterations=None,
            swap_memory=False,
            time_major=False,
            scope=name + str(idx)
            )
        prev_layer = lrelu(prev_layer)
    return prev_layer

def prepare_inputs(prev_layer, prev_layer_size, size, layers, bidirectional=False):
    
    layer_reshaped = tf.reshape(prev_layer, [-1, prev_layer_size])
    
    
    '''
        self.sent_input = tf.reshape(self.wds_usrs,[-1, self.max_sent_len, self.size_wd + self.size_usr])
        
        self.sent_init_states = [
                (layers.dense(self.sent_input, size_enc//2), layers.dense(self.sent_input, size_enc//2))
                for _ in range(layers_enc)
    '''

class Model():
    def __init__(self, layers_enc=1, layers_ctx=1, layers_ctx_2=1,
                 layers_dec=1, layers_dec_2=1,
                 size_usr = 32, size_wd = 32,
                 size_enc=32, size_ctx=32, size_ctx_2=32, 
                 size_dec=32,size_dec_2=32,
                 num_wds = 8192,num_usrs = 1000, weight_decay = 1e-3):
        self.layers_enc = layers_enc
        self.layers_ctx = layers_ctx
        self.layers_dec = layers_dec
        self.size_usr = size_usr
        self.size_wd = size_wd
        self.size_enc = size_enc
        self.size_ctx = size_ctx
        self.size_ctx_2 = size_ctx_2
        self.size_dec = size_dec
        self.num_wds = num_wds
        self.num_srs = num_usrs
        self.size_dec_2 = size_dec_2
        self.layers_dec_2 = layers_dec_2
        
        #Dimension orders: Batch, messages, words, embedding
        
        self.learning_rate = tf.placeholder(tf.float32, shape=(None))
        
        self.wd_ind = tf.placeholder(tf.int32, shape=(None,None, None))
        self.usr_ind = tf.placeholder(tf.int32, shape=(None,None))
        
        self.conv_lengths = tf.placeholder(tf.int32, shape=(None))
        self.sentence_lengths = tf.placeholder(tf.int32, shape=(None,None))
        
        self.batch_size = tf.placeholder(tf.int32, shape=(None))
        self.max_sent_len = tf.placeholder(tf.int32, shape=(None))
        self.max_conv_len = tf.placeholder(tf.int32, shape=(None))
        self.conv_lengths_flat = tf.reshape(self.conv_lengths, [-1])
        self.sentence_lengths_flat = tf.reshape(self.sentence_lengths, [-1])
        
        self.max_sentence_length = tf.reduce_max(self.sentence_lengths)
        
        self.mask_ctx = tf.cast(tf.sequence_mask(self.conv_lengths_flat), tf.float32)
        self.mask_ctx_expanded = tf.reshape(self.mask_ctx, [self.batch_size, self.max_conv_len])
        self.mask = tf.cast(tf.sequence_mask(self.sentence_lengths_flat), tf.float32)
        self.mask_expanded = tf.reshape(self.mask, [self.batch_size, self.max_conv_len, self.max_sent_len])
        self.mask_decode = self.mask_expanded[:,1:,1:]
        self.mask_flat_decode = tf.reshape(self.mask_decode, [self.batch_size * (self.max_conv_len-1), self.max_sent_len-1])
        
        self.wd_mat = tf.Variable(1e-3*tf.random_normal([num_wds, size_wd]))
        self.usr_mat = tf.Variable(1e-3*tf.random_normal([num_usrs, size_usr]))
        
        self.wd_emb = tf.nn.embedding_lookup(self.wd_mat, self.wd_ind)
        self.usr_emb = tf.nn.embedding_lookup(self.usr_mat, self.usr_ind)
        self.usr_emb_flat = tf.reshape(self.usr_emb, [-1, size_usr])
        self.usr_emb_decode = tf.concat((self.usr_emb[:,1:,:], self.usr_emb[:,-2:-1,:]), 1)
        self.usr_emb_decode_flat = tf.reshape(self.usr_emb_decode, [-1, size_usr])
        self.usr_emb_expanded = tf.tile(tf.expand_dims(self.usr_emb, 2),[1,1,self.max_sent_len,1])
        self.ctx_input_init = tf.reduce_mean(self.usr_emb, 1)
        #self.dec_input_init = tf.reshape(self.usr_emb, [-1, self.size_usr])
        
        self.wds_usrs = tf.concat((self.wd_emb, self.usr_emb_expanded), 3)
        self.mask_expanded_wds_usrs = tf.tile(tf.expand_dims(self.mask_expanded, -1), [1,1,1, self.size_wd+self.size_usr])
        self.wds_usrs = self.wds_usrs * self.mask_expanded_wds_usrs
        self.sent_rnns = [[GRUCell(size_enc//2, kernel_initializer=tf.contrib.layers.xavier_initializer()),
                           GRUCell(size_enc//2, kernel_initializer=tf.contrib.layers.xavier_initializer())]
            for _ in range(layers_enc)]
        self.ctx_rnns = [GRUCell(size_ctx, kernel_initializer=tf.contrib.layers.xavier_initializer())
            for _ in range(layers_ctx)]
        self.ctx_rnns_2 = [GRUCell(size_ctx_2, kernel_initializer=tf.contrib.layers.xavier_initializer())
            for _ in range(layers_ctx_2)]
        self.dec_rnns = [GRUCell(size_dec, kernel_initializer=tf.contrib.layers.xavier_initializer())
            for _ in range(layers_dec)]
        self.dec_rnns_2 = [GRUCell(size_dec_2, kernel_initializer=tf.contrib.layers.xavier_initializer())
            for _ in range(layers_dec_2)]
        
        self.sent_input = tf.reshape(self.wds_usrs,[-1, self.max_sent_len, self.size_wd + self.size_usr])
        
        self.sent_mask_contributions = tf.reshape(tf.reduce_sum(
                self.mask_expanded_wds_usrs,2),[ -1, self.size_wd + self.size_usr])
        
        self.sent_input_init = tf.reduce_sum(self.sent_input, 1)/self.sent_mask_contributions
        
        self.sent_init_states = [
                (layers.dense(self.sent_input_init, size_enc//2), layers.dense(self.sent_input_init, size_enc//2))
                for _ in range(layers_enc)]
        self.sentence_encs, self.last_layer_enc = encode_sentence(
                x=self.sent_input, num_layers=layers_enc, size = size_enc, 
                lengths=self.sentence_lengths_flat, cells = self.sent_rnns, initial_states = self.sent_init_states)
        
        self.sentence_encs_shaped = tf.reshape(self.sentence_encs, [self.batch_size, self.max_conv_len,self.size_enc])
        
        #sent_enc_mask = tf.tile(tf.expand_dims(self.mask_ctx_expanded, -1), [1, 1, self.size_enc])
        #ctx_input_init = tf.reduce_sum(sentence_encs * sent_enc_mask, 1) / tf.reduce_sum(sent_enc_mask, 1)
        
        self.ctx_init_states = [layers.dense(self.ctx_input_init, size_ctx) for _ in range(layers_ctx)]
        self.context_encs = encode_context(
                x = self.sentence_encs_shaped, num_layers = layers_ctx, size = size_ctx,
                lengths = self.conv_lengths_flat, cells = self.ctx_rnns, initial_states = self.ctx_init_states)

        self.last_layer_enc_attend = tf.reshape(
                self.last_layer_enc, [self.batch_size, self.max_conv_len, self.max_sent_len, self.size_enc])
        
        #self.context_encs starts as [B, C, V]
        #last layer enc starts as [B, C, S, V]
        #Context becomes [B, C, CDuplicate, SDuplicate, V]
        #Last layer enc becomes [B, CDuplicate, C, S, V]
        #Sum over last layer index 4, then 3
        self.context_masked = tf.tile(tf.expand_dims(tf.reduce_max(
                self.mask_expanded, -1), -1), [1, 1, self.size_ctx]) * self.context_encs
        self.last_layer_enc_attend_masked = tf.tile(
                tf.expand_dims(self.mask_expanded, -1), [1, 1, 1,self.size_enc]) * self.last_layer_enc_attend
        self.context_wd_Attention= Attention(
                head=self.context_masked, 
                history=self.last_layer_enc_attend_masked, 
                head_shape = [self.batch_size, self.max_conv_len, self.size_ctx],
                history_shape = [self.batch_size, self.max_conv_len, self.max_sent_len, self.size_enc],
                head_expanded_shape = [self.batch_size, self.max_conv_len, self.max_conv_len, self.max_sent_len, self.size_ctx], 
                history_expanded_shape = [self.batch_size, self.max_conv_len, self.max_conv_len, self.max_sent_len, self.size_enc],
                head_expand_dims = [(2, self.max_conv_len), (3, self.max_sent_len)], 
                history_expand_dims = [(1, self.max_conv_len)], 
                history_rollup_dims = [(2, self.max_conv_len),(3, self.max_sent_len)],
                name='ctx_wd_attn')
        self.ctx_wd_attn = self.context_wd_Attention.attended_history_rolledup
        self.context_encs_with_attn = tf.concat((self.context_encs, self.ctx_wd_attn), 2)
        
        self.ctx_init_states_2 = [layers.dense(self.ctx_input_init, size_ctx_2) for _ in range(layers_ctx_2)]
        self.context_encs_second_rnn = encode_context(
                x = self.context_encs_with_attn, num_layers = layers_ctx_2, size = size_ctx_2,
                lengths = self.conv_lengths_flat, cells = self.ctx_rnns_2,
                initial_states = self.ctx_init_states_2, name = 'ctx_2_')
        
        self.context_enc_rar = tf.concat((self.context_encs_with_attn, self.context_encs_second_rnn), 2)
        self.ctx_for_rec = tf.reshape(self.context_enc_rar, [-1, self.size_ctx + self.size_enc + self.size_ctx_2])
        self.thought_rec = MLP(self.ctx_for_rec, [self.size_enc], self.size_enc, 'thought_rec')
        self.thought_rec_reshaped = tf.reshape(self.thought_rec, [self.batch_size, self.max_conv_len, self.size_enc])
        self.context_encs_final = tf.concat((self.context_enc_rar, self.thought_rec_reshaped), 2)
        
        self.thought_rec_loss = tf.reduce_sum(tf.square(
            tf.stop_gradient(self.sentence_encs_shaped[:,1:,:]) - self.thought_rec_reshaped[:,:-1,:]) * \
                tf.expand_dims(self.mask_ctx_expanded[:,1:],-1)) / tf.reduce_sum(self.mask_ctx_expanded[:,1:]) * 1e-3
        
        
        self.size_ctx_final = self.size_ctx + self.size_enc + self.size_ctx_2 + self.size_enc
        self.dec_inputs_deep = tf.concat(
                (tf.tile(tf.expand_dims(self.context_encs_final, 2),[1,1,self.max_sent_len,1]), self.wds_usrs), 3)
        
        self.dec_inputs = tf.reshape(
                self.dec_inputs_deep,
                [-1, self.max_sent_len,self.size_wd + self.size_usr + self.size_ctx_final])
        
        self.target = self.wd_ind[:,1:, 1:]
        self.target_decode = tf.reshape(self.target, [self.batch_size * (self.max_conv_len - 1), self.max_sent_len-1])
        #self.target_flat = tf.reshape(self.target, [-1])
        
        self.dec_input_init = tf.reshape(self.context_encs_final, 
                [-1, self.size_ctx_final])
        
        self.dec_init_states = [layers.dense(self.dec_input_init, size_dec) for _ in range(layers_dec)]
        
        self.dec_init_states_2 = [layers.dense(self.dec_input_init, size_dec_2) for _ in range(layers_dec_2)]
        
        self.decoder_out = decode(
                x = self.dec_inputs, num_layers=layers_dec, size=size_dec,
                lengths = self.sentence_lengths_flat, cells = self.dec_rnns, initial_states = self.dec_init_states)
        #decoder out is message0: message-1
        self.decoder_deep = tf.reshape(
                self.decoder_out, (self.batch_size, self.max_conv_len, self.max_sent_len, self.size_dec))
        
        #self.context_encs starts as [B, C, V]
        #self.decoder_deep starts as [B,C,S,V]
        #Context becomes [B, CDuplicate, C, SDuplicate, V]
        #Decoder becomes [B, C, CDuplicate, S, V]
        #Sum over last layer index 2
        self.dec_ctx_Attention= Attention(
                head=self.decoder_deep, 
                history=self.context_encs_final, 
                head_shape = [self.batch_size, self.max_conv_len, self.max_sent_len, self.size_dec],
                history_shape = [self.batch_size, self.max_conv_len, self.size_ctx_final],
                head_expanded_shape = [self.batch_size, self.max_conv_len, self.max_conv_len, self.max_sent_len, self.size_dec], 
                history_expanded_shape = [self.batch_size, self.max_conv_len, self.max_conv_len, self.max_sent_len, self.size_ctx_final],
                head_expand_dims = [(2, self.max_conv_len)], 
                history_expand_dims = [(1, self.max_conv_len),(3, self.max_sent_len)], 
                history_rollup_dims = [(2, self.max_conv_len)],
                name='dec_ctx_attn')
        self.dec_ctx_attn = self.dec_ctx_Attention.attended_history_rolledup
        self.decoder_with_attn = tf.concat((self.decoder_deep, self.dec_ctx_attn), -1)
        if 1:
            self.decoder_for_attn = tf.reshape(
                    self.decoder_with_attn, 
                    (self.batch_size*self.max_conv_len*self.max_sent_len,
                     self.size_dec + self.size_ctx_final))
            self.reconstruct = layers.dense(
                    lrelu(layers.dense(self.decoder_for_attn, self.size_wd, name='rec_layer')),
                    self.size_wd, name='rec_layer2')
            self.reconstruct_for_penalty = tf.reshape(
                    self.reconstruct, 
                    (self.batch_size,self.max_conv_len,self.max_sent_len,self.size_wd))[:,:-1,:-1,:]
            self.wd_emb_for_reconstruct = tf.stop_gradient(
                    tf.reshape(self.wd_emb,
                    (self.batch_size,self.max_conv_len,self.max_sent_len,self.size_wd))[:,1:,1:,:]
                    )
            self.reconstruct_loss = tf.reduce_sum(tf.square(
                    self.reconstruct_for_penalty - self.wd_emb_for_reconstruct) * 
                    tf.tile(tf.expand_dims(self.mask_decode,-1), [1, 1, 1, self.size_wd])) / tf.reduce_sum(
                            self.mask_decode) * 1e-6
            self.reconstruct_shaped = tf.reshape(
                    self.reconstruct,
                    (self.batch_size*self.max_conv_len,self.max_sent_len,self.size_wd))
            self.decoder_with_attn_shaped = tf.reshape(
                    self.decoder_with_attn,
                    (self.batch_size*self.max_conv_len, self.max_sent_len, self.size_dec + self.size_ctx_final))
            self.decoder_second_rnn = decode(
                    x = self.decoder_with_attn_shaped, num_layers = layers_dec_2, size = size_dec_2,
                    lengths = self.sentence_lengths_flat, cells = self.dec_rnns_2,
                    initial_states = self.dec_init_states_2, name = 'dec_2_')
            self.decoder_attn_and_second_rnn = tf.concat(
                    (self.decoder_with_attn_shaped, self.decoder_second_rnn, self.reconstruct_shaped), -1)
            self.size_dec_final = size_dec + self.size_ctx_final + size_dec_2 + self.size_wd
            self.decoder_attn_and_second_rnn_reshaped = tf.reshape(
                    self.decoder_attn_and_second_rnn, 
                    (self.batch_size, self.max_conv_len, self.max_sent_len, self.size_dec_final))
            self.decoder_out_for_ppl = tf.reshape(self.decoder_attn_and_second_rnn_reshaped[:,:-1,:,:],(-1, self.size_dec_final))
        else:
            self.size_dec_final = size_dec + self.size_ctx_final
            self.decoder_out_for_ppl = tf.reshape(self.decoder_with_attn[:,:-1,:,:],(-1, self.size_dec_final))
        #self.dec_raw = layers.dense(self.decoder_out_for_ppl, self.num_wds, name='output_layer')
        self.dec_raw = MLP(
                self.decoder_out_for_ppl, hiddens = [self.size_dec, self.size_dec],
                output_size = self.num_wds, name = 'output_layer')
        self.dec_relu = lrelu(self.dec_raw)
        self.dec_avg = tf.reduce_sum(
                self.dec_relu * tf.expand_dims(tf.reshape(
                        self.mask_expanded[:,1:,:], [-1]),-1), 0)/tf.reduce_sum(self.mask_decode)
        self.overuse_penalty = tf.reduce_mean(tf.pow(self.dec_avg, 2))
        self.dec_relu_shaped = tf.reshape(
                self.dec_relu, [self.batch_size *  (self.max_conv_len-1), self.max_sent_len, self.num_wds])[:,:-1,:]        
        #self.labels = tf.one_hot(indices = self.target_flat, depth = self.num_wds)
        self.ppl_loss_masked = self.ppl_loss_raw = tf.contrib.seq2seq.sequence_loss(
            logits = self.dec_relu_shaped,
            targets = self.target_decode,
            weights = self.mask_flat_decode,
            average_across_timesteps=False,
            average_across_batch=False,
            softmax_loss_function=None,
            name=None
        )
        self.ppl_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(
                self.ppl_loss_masked, 2.0), 1), 0)/tf.reduce_sum(self.mask_flat_decode)/10
        self.greedy_words, self.greedy_pre_argmax = self.decode_greedy(num_layers=layers_dec, 
                max_length = 30, start_token = dataset.index_word(START))
        
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.tfvars = tf.trainable_variables()
        self.weight_norm = tf.reduce_mean([tf.reduce_sum(tf.square(var)) for var in self.tfvars])*weight_decay
        loss = (self.ppl_loss + self.weight_norm + self.reconstruct_loss + self.thought_rec_loss + self.overuse_penalty)
        gvs = optimizer.compute_gradients(self.ppl_loss)
        self.grad_norm_ppl = tf.reduce_mean([tf.reduce_mean(tf.square(grad)) for grad, var in gvs if grad is not None])
        gvs = optimizer.compute_gradients(self.reconstruct_loss)
        self.grad_norm_rec_wd = tf.reduce_mean([tf.reduce_mean(tf.square(grad)) for grad, var in gvs if grad is not None])
        gvs = optimizer.compute_gradients(self.thought_rec_loss)
        self.grad_norm_rec_thought = tf.reduce_mean([tf.reduce_mean(tf.square(grad)) for grad, var in gvs if grad is not None])
        gvs = optimizer.compute_gradients(self.overuse_penalty)
        self.grad_norm_overuse = tf.reduce_mean([tf.reduce_mean(tf.square(grad)) for grad, var in gvs if grad is not None])
        gvs = optimizer.compute_gradients(loss)
        self.grad_norm_total = tf.reduce_mean([tf.reduce_mean(tf.square(grad)) for grad, var in gvs if grad is not None])
        clip_norm = 1
        clip_single = .001
        capped_gvs = [(tf.clip_by_value(grad, -1*clip_single,clip_single), var)
                      for grad, var in gvs if grad is not None]
        capped_gvs = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in capped_gvs if grad is not None]
        self.optimizer = optimizer.apply_gradients(capped_gvs)
        
    def decode_greedy(self, num_layers, max_length, start_token):
        ctx_flat = tf.reshape(self.context_encs_final, [-1, self.size_ctx_final])
        words = []
        pre_argmax_values = []
        start_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.wd_mat, tf.expand_dims(start_token, -1)), 0)
        start_embedding = tf.tile(start_embedding, [self.batch_size, self.max_conv_len, 1])
        start_usrs = tf.concat((start_embedding, self.usr_emb_decode), 2)
        prev_layer = tf.concat((self.context_encs_final, start_usrs), -1)
        context_encs_final = self.context_encs_final[0,:,:]
        #prev_state = [tf.zeros(self.size_dec) for _ in range(num_layers)]
        #prev_state = [tf.zeros((self.batch_size*self.max_conv_len, self.size_dec)) for _ in range(num_layers)]
        prev_state = self.dec_init_states
        prev_state_2 = self.dec_init_states_2
        prev_layer = tf.reshape(
                prev_layer, [-1, self.size_wd + self.size_usr + self.size_ctx_final])
        for _ in range(max_length):
            for idx in range(num_layers):
                prev_layer,prev_state[idx] =self.dec_rnns[idx](inputs = prev_layer, state = prev_state[idx])
                
            
            dec_ctx_Attention= Attention(
                head=prev_layer, 
                history=context_encs_final, 
                head_shape = [self.max_conv_len, self.size_dec],
                history_shape = [self.max_conv_len, self.size_ctx_final],
                head_expanded_shape = [self.max_conv_len, self.max_conv_len, self.size_dec], 
                history_expanded_shape = [self.max_conv_len, self.max_conv_len, self.size_ctx_final],
                head_expand_dims = [(1, self.max_conv_len)], 
                history_expand_dims = [(0, self.max_conv_len)], 
                history_rollup_dims = [(1, self.max_conv_len)],
                name='dec_ctx_attn')
            dec_ctx_attn = dec_ctx_Attention.attended_history_rolledup
            prev_layer = tf.concat((prev_layer,dec_ctx_attn),-1)
            pre_second_rnn = prev_layer
            
            reconstruct = layers.dense(
                    lrelu(layers.dense(pre_second_rnn, self.size_wd, name='rec_layer', reuse=True)),
                    self.size_wd, name='rec_layer2', reuse=True)
            
<<<<<<< HEAD
            for idx in range(self.layers_dec_2):
                prev_layer,prev_state_2[idx] =self.dec_rnns_2[idx](inputs = prev_layer, state = prev_state_2[idx])
            prev_layer = tf.concat((pre_second_rnn,prev_layer, reconstruct),-1)
            #prev_layer = layers.dense(prev_layer, self.num_wds, name='output_layer', reuse=True)
            prev_layer= MLP(
                prev_layer, hiddens = [self.size_dec, self.size_dec],
                output_size = self.num_wds, name = 'output_layer', reuse=True)
            next_words = tf.argmax(prev_layer, -1)
            words.append(next_words)
            pre_argmax_values.append(prev_layer)
            wd_embeddings = tf.nn.embedding_lookup(self.wd_mat, next_words)
            wds_usrs = tf.concat((wd_embeddings, self.usr_emb_decode_flat), -1)
            prev_layer = tf.concat((ctx_flat, wds_usrs), -1)
        return words, pre_argmax_values
=======
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
    def __init__(self, size_context, max_turns_allowed, num_layers = 1):
        NN.Module.__init__(self)
        self._size_context = size_context
        self._max_turns_allowed = max_turns_allowed
        self._num_layers = num_layers
        self.softmax = NN.Softmax()
        self.F = NN.Sequential(
            NN.Linear(size_context * 2, size_context),
            NN.LeakyReLU(),
            NN.Linear(size_context, size_context),
            NN.LeakyReLU(),
            NN.Linear(size_context, 1))
        init_weights(self.F)

    def forward(self, sent_encodings, turns):
        batch_size, num_turns, size_context = sent_encodings.size()
        sent_encodings = cuda(sent_encodings)
        max_turns_allowed = self._max_turns_allowed
        num_layers = self._num_layers
        attn_heads = sent_encodings.unsqueeze(1).expand(batch_size, num_turns, num_turns, size_context)
        ctx_to_attend = sent_encodings.unsqueeze(2).expand(batch_size, num_turns, num_turns, size_context)
        head_and_ctx = T.cat((attn_heads, ctx_to_attend),3)
        attn_raw = self.F(head_and_ctx.view(batch_size *num_turns* num_turns, size_context + size_context))
        attn_raw = attn_raw.view(batch_size, num_turns, num_turns, 1)
        
        
        size = attn_raw.size()
        mask = T.ones(*size)
    
        if isinstance(turns, T.LongTensor):
            for i_b in range(size[0]):
                for i_head in range(size[1]):
                    for i_sent in range(size[2]):
                        if ((turns[i_b] < i_head)
                            or (i_head <= i_sent)
                            or (turns[i_b] <= i_sent)):
                                mask[i_b, i_head, i_sent] = 0
                                
        mask = tovar(mask)
        attn_raw = attn_raw * mask
        # FIXME: variable length softmax
        attn_raw = weighted_softmax(
                attn_raw.view(batch_size * num_turns, num_turns),
                mask.view(batch_size*num_turns, num_turns)).view(
                        batch_size, num_turns, num_turns, 1)

        at_weighted_sent = attn_raw * ctx_to_attend 
        at_weighted_sent = at_weighted_sent.sum(2)
        return at_weighted_sent, attn_raw

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
            in_size = size_usr + size_wd + context_size*2 + size_sentence + size_usr
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
        self.softmax = HierarchicalLogSoftmax(state_size*2, np.int(np.sqrt(num_words)), num_words)
        init_lstm(self.rnn)
        if self._attention_enabled:
            self.attention = Attention(state_size, 100,num_layers = 1)

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
        embed = embed.permute(1, 0, 2).contiguous()
        
        attn, attn_weight = self.attention(embed, length.contiguous().view(-1))
        
        embed = T.cat((embed, attn),2)
        embed = embed.view(batch_size, maxlenbatch, maxwordsmessage, state_size*2)

        if wd_target is None:
            out = self.softmax(embed.view(-1, state_size * 2))
            out = out.view(batch_size, maxlenbatch, -1, self._num_words)
            log_prob = None
        else:
            target = T.cat((wd_target[:, :, 1:], tovar(T.zeros(batch_size, maxlenbatch, 1)).long()), 2)
            out = self.softmax(embed.view(-1, state_size * 2), target.view(-1))
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
        embed = embed.permute(1, 0, 2).contiguous()
        attn = self.attention(embed, False)
        
        embed = T.cat((embed, attn),2)
        #embed = embed.view(batch_size, -1, maxwordsmessage, self._state_size*2)
        embed = embed.view(batch_size, self._state_size*2)
        out = self.softmax(embed.squeeze())
        val, indexes = out.topk(n, 1)
        return val, indexes, current_state

    def get_next_word(self, embed_seq, init_state):
        """

        :param embed_seq: batch_size x (usr_emb_size + w_emb_size + context_emb_size)
        :param current_state: num_layers * num_directions x batch_size x hidden_size
        :param n: int, return top n words.
        :return: batch_size x n
        """
        embed_seq = cuda(embed_seq)

        num_decoded, num_turns_decode, state_size = embed_seq.size()
        embed, current_state = self.rnn(embed_seq, init_state)
        embed = embed.permute(1, 0, 2).contiguous()
        attn = self.attention(embed, False)
        
        embed = T.cat((embed, attn),2)
        #embed = embed.view(batch_size, -1, maxwordsmessage, self._state_size*2)
        embed = embed.view(num_decoded, num_turns_decode, self._state_size*2)
        embed = embed[-1,:,:].contiguous()
        out = self.softmax(embed.squeeze())
        val, indexes = out.topk(1, 1)
        return indexes

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
        init_state = cuda((lstm_h, lstm_c))

        # Initial word of response : Start token
        init_word = tovar(T.LongTensor(batch_size).fill_(dataset.index_word(START)))
        # End of generated sentence : EOS token
        stop_word = cuda(T.LongTensor(batch_size).fill_(dataset.index_word(EOS)))

        current_w = init_word
        output = current_w.data.unsqueeze(1)
        init_seq = 0
        while not stop_word.equal(current_w.data.squeeze()) and output.size(1) < max_len_generated:
            current_w_emb = word_emb(current_w.squeeze())
            if init_seq == 0:
                init_seq = 1
                embed_seq = T.cat((usr_emb, current_w_emb, context_encodings), 1).unsqueeze(0).contiguous()
            else:
                X_i = T.cat((usr_emb, current_w_emb, context_encodings), 1).contiguous()
                embed_seq = T.cat((embed_seq, X_i.unsqueeze(0)),0)
            current_w = self.get_next_word(embed_seq,init_state)
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
>>>>>>> ab3bd387d51524c6d786a6a1bc6f0a837c68e09c

        
 
parser = argparse.ArgumentParser(description='Ubuntu Dialogue dataset parser')
parser.add_argument('--dataroot', type=str,default='OpenSubtitles-dialogs-small', help='Root of the data downloaded from github')
parser.add_argument('--metaroot', type=str, default='opensub', help='Root of meta data')
#796
parser.add_argument('--vocabsize', type=int, default=39996, help='Vocabulary size')
parser.add_argument('--gloveroot', type=str,default='glove', help='Root of the data downloaded from github')
parser.add_argument('--outputdir', type=str, default ='outputs',help='output directory')
parser.add_argument('--logdir', type=str, default='logs', help='log directory')
parser.add_argument('--layers_enc', type=int, default=3)
parser.add_argument('--layers_ctx', type=int, default=2)
parser.add_argument('--layers_dec', type=int, default=2)
parser.add_argument('--layers_ctx_2', type=int, default=1)
parser.add_argument('--layers_dec_2', type=int, default=1)
parser.add_argument('--size_enc', type=int, default=128)
parser.add_argument('--size_attn', type=int, default=0)
parser.add_argument('--size_ctx', type=int, default=256)
parser.add_argument('--size_dec', type=int, default=512)
parser.add_argument('--size_ctx_2', type=int, default=256)
parser.add_argument('--size_dec_2', type=int, default=512)
parser.add_argument('--size_usr', type=int, default=16)
parser.add_argument('--size_wd', type=int, default=64)
parser.add_argument('--weight_decay', type=float, default=1e-7)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--gradclip', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--modelname', type=str, default = '')
parser.add_argument('--modelnamesave', type=str, default='')
parser.add_argument('--modelnameload', type=str, default='')
parser.add_argument('--loaditerations', type=int, default=0)
parser.add_argument('--max_sentence_length_allowed', type=int, default=16)
parser.add_argument('--max_turns_allowed', type=int, default=8)
parser.add_argument('--num_loader_workers', type=int, default=4)
parser.add_argument('--adversarial_sample', type=int, default=0)
parser.add_argument('--emb_gpu_id', type=int, default=0)
parser.add_argument('--ctx_gpu_id', type=int, default=0)
parser.add_argument('--enc_gpu_id', type=int, default=0)
parser.add_argument('--dec_gpu_id', type=int, default=0)
parser.add_argument('--lambda_repetitive', type=float, default=.003)
parser.add_argument('--lambda_reconstruct', type=float, default=.1)
parser.add_argument('--hidden_width', type=int, default=1)
parser.add_argument('--server', type=int, default=0)
parser.add_argument('--plot_attention', action='store_true', help='Plot attention')

args = parser.parse_args()
if args.server == 1:
    args.dataroot = '/misc/vlgscratch4/ChoGroup/gq/data/OpenSubtitles/OpenSubtitles-dialogs/'
    args.metaroot = 'opensub'
    args.logdir = '/home/qg323/lee/'
print(args)

datasets = []
dataloaders = []
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

print('Checking consistency...')
for dataset in datasets:
    assert all(w1 == w2 for w1, w2 in zip(datasets[0].vocab, dataset.vocab))
    assert all(u1 == u2 for u1, u2 in zip(datasets[0].users, dataset.users))

dataloader = round_robin_dataloader(dataloaders)


try:
    os.mkdir(args.logdir)
except:
    pass

try:
    os.mkdir("plot")
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

train_writer = tf.summary.FileWriter(log_train)

adv_min_itr = 1000


vcb = dataset.vocab
usrs = dataset.users
num_usrs = len(usrs)
vcb_len = len(vcb)
lr = args.lr
num_words = vcb_len
size_usr = args.size_usr
size_wd = args.size_wd
size_enc = args.size_enc
size_ctx = args.size_ctx
size_dec = args.size_dec
size_ctx_2 = args.size_ctx_2
size_dec_2 = args.size_dec_2
size_attn = args.size_attn = 10 #For now hard coding to 10.
layers_enc = args.layers_enc
layers_ctx = args.layers_ctx
layers_dec = args.layers_dec
layers_ctx_2 = args.layers_ctx_2
layers_dec_2 = args.layers_dec_2
weight_decay = args.weight_decay

model = Model(layers_enc=layers_enc, 
              layers_ctx=layers_ctx, layers_ctx_2=layers_ctx_2, 
              layers_dec=layers_dec,layers_dec_2=layers_dec_2,
              size_usr = size_usr, size_wd = size_wd,
              size_enc=size_enc, 
              size_ctx=size_ctx, size_ctx_2=size_ctx_2, 
              size_dec=size_dec,size_dec_2=size_dec_2,
              num_wds = num_words+1,num_usrs = num_usrs+1, weight_decay = weight_decay)


start = dataset.index_word(START)
eos = dataset.index_word(EOS)
unk = dataset.index_word(UNKNOWN)

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
sess = tf.Session()
sess.run(tf.global_variables_initializer())
if modelnameload:
    if len(modelnameload) > 0:
        pass
        user_emb = T.load('%s-user_emb-%08d' % (modelnameload, args.loaditerations))
        word_emb = T.load('%s-word_emb-%08d' % (modelnameload, args.loaditerations))
        enc = T.load('%s-enc-%08d' % (modelnameload, args.loaditerations))
        context = T.load('%s-context-%08d' % (modelnameload, args.loaditerations))
        decoder = T.load('%s-decoder-%08d' % (modelnameload, args.loaditerations))
adv_style = 0
scatter_entropy_freq = 200
time_train = 0
time_decode = 0
first_sample = 1
while True:
    epoch += 1
    for item in dataloader:
        start_train = time.time()
        if itr % scatter_entropy_freq == 0:
            adv_style = 1 - adv_style
            #adjust_learning_rate(opt, args.lr / np.sqrt(1 + itr / 10000))
        itr += 1
        repeat = 1
        turns, sentence_lengths_padded, speaker_padded, \
            addressee_padded, words_padded, words_reverse_padded = item
<<<<<<< HEAD
        if first_sample:
            turns_first, sentence_lengths_padded_first, speaker_padded_first, \
                addressee_padded_first, words_padded_first, words_reverse_padded_first = \
                    turns, sentence_lengths_padded, speaker_padded, \
                    addressee_padded, words_padded, words_reverse_padded
            first_sample = 0
        if repeat:
            turns, sentence_lengths_padded, speaker_padded, \
                    addressee_padded, words_padded, words_reverse_padded = \
                        turns_first, sentence_lengths_padded_first, speaker_padded_first, \
                        addressee_padded_first, words_padded_first, words_reverse_padded_first
            
        if args.plot_attention:
            list_stringify_sentence = batch_to_string_sentence(words_padded, sentence_lengths_padded, dataset)
        else:
            list_stringify_sentence = None

        wds = sentence_lengths_padded.sum()
        max_wds = args.max_turns_allowed * args.max_sentence_length_allowed
        if sentence_lengths_padded.shape[1] < 2 and repeat == 0:
            continue
        if (wds > max_wds * .8 or wds < max_wds * .05) and repeat == 0:
            continue
        
        
        batch_size = turns.shape[0]
        max_turns = words_padded.shape[1]
        max_words = words_padded.shape[2]
        #batch, turns in a sample, words in a message
        feed_dict = {
                model.learning_rate: lr,
                model.wd_ind: words_padded,
                model.usr_ind: speaker_padded,
                model.conv_lengths: turns,
                model.sentence_lengths: sentence_lengths_padded,
                model.batch_size: batch_size,
                model.max_sent_len: max_words,
                model.max_conv_len: max_turns
                }
        
        
        
        if itr % 10 == 0:
            _, loss, weight_norm, rec_loss_wd, rec_loss_thought, overuse_penalty, \
                grad_norm_ppl, grad_norm_rec_wd, grad_norm_rec_thought, grad_norm_overuse, grad_norm_total =  sess.run(
                    [model.optimizer,model.ppl_loss, model.weight_norm, model.reconstruct_loss, model.thought_rec_loss,
                     model.overuse_penalty,
                     model.grad_norm_ppl, model.grad_norm_rec_wd, model.grad_norm_rec_thought, 
                     model.grad_norm_overuse,model.grad_norm_total],
                    feed_dict=feed_dict)
=======
        batch_word_str = batch_to_string_sentence(words_padded, sentence_lengths_padded, dataset)
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
            scale = float(np.exp(-np.random.uniform(5, 6)))
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
            scale = float(np.exp(-np.random.uniform(5, 6)))
            wds_adv, usrs_adv, enc_adv, loss_adv = adversarial_encodings_wds_usrs(encodings, batch_size, 
                    wds_b,usrs_b,max_turns, context, turns, 
                    sentence_lengths_padded, words_padded, decoder,
                    usr_std, wd_std, sent_std, wds_h, scale=scale, style=adv_style)
            wds_b = tovar((wds_b + tovar(wds_adv)).data)
            usrs_b = tovar((usrs_b + tovar(usrs_adv)).data)
            encodings = tovar((encodings + tovar(enc_adv).view_as(encodings)).data)
        encodings = encodings.view(batch_size, max_turns, -1)
        #attn = attention(encodings)
        ctx, _ = context(encodings, turns, sentence_lengths_padded, wds_h.contiguous(), usrs_b, plot=True, sentence_string=batch_word_str)
        if itr % 10 == 7 and args.adversarial_sample == 1:
            scale = float(np.exp(-np.random.uniform(5, 6)))
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
>>>>>>> ab3bd387d51524c6d786a6a1bc6f0a837c68e09c
            train_writer.add_summary(
                    tf.Summary(
                        value=[
                            tf.Summary.Value(tag='perplexity', simple_value=np.exp(np.min((8, tonumpy(loss))))),
                            tf.Summary.Value(tag='loss', simple_value=loss),
                            tf.Summary.Value(tag='weight_norm', simple_value=weight_norm),
                            tf.Summary.Value(tag='rec_loss_wd', simple_value=rec_loss_wd),
                            tf.Summary.Value(tag='rec_loss_thought', simple_value=rec_loss_thought),
                            tf.Summary.Value(tag='overuse_penalty', simple_value=overuse_penalty),
                            tf.Summary.Value(tag='grad_norm_ppl', simple_value=grad_norm_ppl),
                            tf.Summary.Value(tag='grad_norm_rec_wd', simple_value=grad_norm_rec_wd),
                            tf.Summary.Value(tag='grad_norm_rec_thought', simple_value=grad_norm_rec_thought),
                            tf.Summary.Value(tag='grad_norm_overuse', simple_value=grad_norm_overuse),
                            tf.Summary.Value(tag='grad_norm_total', simple_value=grad_norm_total),
                            ]
                        ),
                    itr
                    )
        else:
            sess.run(model.optimizer,feed_dict=feed_dict)
            
        if np.isnan(grad_norm):
            breakhere = 1
        if itr % 100 == 0:
            a = 2
        if itr % 100 == 0:
            greedy_responses, greedy_values = sess.run(
                    [model.greedy_words, model.greedy_pre_argmax],
                    feed_dict = feed_dict)
            print_greedy_decode(words_padded, greedy_responses, greedy_values)
        
        if itr % 10 == 0:
            print('Epoch', epoch, 'Iteration', itr, 'Loss', tonumpy(loss), 'PPL', np.exp(np.min((10, tonumpy(loss)))))
