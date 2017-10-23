
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

    def forward(self, context_encodings, wd_emb, usr_emb, length, wd_target=None):
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

    def getNBestNextWords(self, embed_seq, current_state, n=1):
        """

        :param embed_seq: batch_size x (usr_emb_size + w_emb_size + context_emb_size)
        :param current_state: num_layers * num_directions x batch_size x hidden_size
        :param n: int, return top n words.
        :return: batch_size x n
        """
        batch_size = embed_seq.size(0)
        embed, current_state = self.rnn(embed_seq.unsqueeze(0), current_state)
        embed = embed.permute(1, 0, 2).contiguous().view(batch_size, self._state_size)
        out = self.softmax(embed.squeeze())
        val, indexes = out.topk(n, 1)
        return val, indexes, current_state

    def greedyGenerate(self, context_encodings, usr_emb, word_emb, dataset):
        """
        How to require_grad=False ?
        :param context_encodings: (batch_size x context_size)
        :param word_emb:  idx to vector word embedder.
        :param usr_emb: (batch_size x usr_emb_size)
        :return: response : (batch_size x max_response_length)
        """

        num_layers = self._num_layers
        state_size = self._state_size
        batch_size = context_encodings.size(0)
        lstm_h = tovar(T.zeros(num_layers, batch_size, state_size))
        lstm_c = tovar(T.zeros(num_layers, batch_size, state_size))
        print(lstm_h.size())
        current_state = (lstm_h, lstm_c)

        # Initial word of response : Start token
        init_word = tovar(T.LongTensor(batch_size).fill_(dataset.index_word(START)))
        # End of generated sentence : EOS token
        stop_word = T.LongTensor(batch_size).fill_(dataset.index_word(EOS))

        current_w = init_word
        output = current_w.data.unsqueeze(1)

        while not stop_word.equal(current_w.data.squeeze()) and output.size(1) < 10:
            current_w_emb = word_emb(current_w.squeeze())
            embed_seq = T.cat((usr_emb, current_w_emb, context_encodings), 1)

            _, current_w, current_state= self.getNBestNextWords(embed_seq, current_state)
            output = T.cat((output, current_w.data), 1)

        return output

    def viterbiGenerate(self, context_encodings, usr_emb, word_emb, dataset, beam_size=13):
        """
        :param context_encodings: (batch_size x context_size)
        :param usr_emb: (batch_size x usr_emb_size)
        :param word_emb:  idx to vector word embedder.
        :param dataset:  dataset to get EOS/START tokens ids
        :param beam_size:  width of beam search.
        :return: response : (batch_size x max_response_length)
        """

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

        s_idx = 0

        # First step from START to first probable words :

        lstm_h = tovar(T.zeros(num_layers, batch_size, state_size))
        lstm_c = tovar(T.zeros(num_layers, batch_size, state_size))

        current_w_emb = word_emb(initial_word.unsqueeze(1))
        embed_seq = T.cat((usr_emb, current_w_emb, context_encodings), 2)
        transition_probabilities, transition_index, current_state = self.getNBestNextWords(embed_seq.squeeze(), (lstm_h, lstm_c), beam_size)

        s_idx_w_idx[0,s_idx] = transition_index
        s_idx_w_idx_logproba[s_idx] = transition_probabilities
        s_idx_w_idx_lstm_h = current_state[0].unsqueeze(2).expand_as(s_idx_w_idx_lstm_h).contiguous()
        s_idx_w_idx_lstm_c = current_state[1].unsqueeze(2).expand_as(s_idx_w_idx_lstm_h).contiguous()

        usr_emb = usr_emb.expand(usr_emb.size(0), beam_size, usr_emb.size(2))
        context_encodings = context_encodings.expand(context_encodings.size(0), beam_size, context_encodings.size(2))

        while s_idx < 10:
            current_w_emb = word_emb(s_idx_w_idx[0,s_idx])
            embed_seq = T.cat((usr_emb, current_w_emb, context_encodings), 2)
            transition_probabilities, transition_index,current_state = self.getNBestNextWords(embed_seq.view(-1, embed_seq.size(2)), (s_idx_w_idx_lstm_h.view(num_layers, -1, state_size), s_idx_w_idx_lstm_c.view(num_layers, -1, state_size)), beam_size)

            transition_probabilities = transition_probabilities.view(batch_size, beam_size, beam_size)
            transition_index = transition_index.view(batch_size, beam_size, beam_size)

            transition_probabilities.add_(s_idx_w_idx_logproba[s_idx].unsqueeze(2).expand_as(transition_probabilities))

            # From transition probability to overall probability for the path.
            # We need the best beam_size out of beam_size x beam_size in a pytorch nice way.
            s_idx += 1
            sys.exit(1)

        pass



parser = argparse.ArgumentParser(description='Ubuntu Dialogue dataset parser')
parser.add_argument('--dataroot', type=str,default='ubuntu', help='Root of the data downloaded from github')
parser.add_argument('--outputdir', type=str, default ='outputs',help='output directory')
parser.add_argument('--logdir', type=str, default='logs', help='log directory')
parser.add_argument('--encoder_layers', type=int, default=1)
parser.add_argument('--decoder_layers', type=int, default=1)
parser.add_argument('--size_context', type=int, default=12)
parser.add_argument('--size_sentence', type=int, default=6)
parser.add_argument('--decoder_size_sentence', type=int, default=4)
parser.add_argument('--size_usr', type=int, default=10)
parser.add_argument('--size_wd', type=int, default=8)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--gradclip', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--modelname', type=str, default = '')
parser.add_argument('--modelnamesave', type=str, default='')
parser.add_argument('--modelnameload', type=str, default='')
parser.add_argument('--loaditerations', type=int, default=0)
args = parser.parse_args()

dataset = UbuntuDialogDataset('ubuntu', 'wordcount.pkl', 'usercount.pkl')
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

user_emb = cuda(NN.Embedding(num_usrs+1, size_usr, padding_idx = 0))
word_emb = cuda(NN.Embedding(vcb_len+1, size_wd, padding_idx = 0))
enc = cuda(Encoder(size_usr, size_wd, size_sentence, num_layers = 1))
context = cuda(Context(size_sentence, size_context, num_layers = 1))
decoder = cuda(Decoder(size_usr, size_wd, size_context, num_words+1,
                       decoder_size_sentence))

params = sum([list(m.parameters()) for m in [user_emb, word_emb, enc, context, decoder]], [])
opt = T.optim.Adam(params, lr=args.lr)


dataloader = UbuntuDialogDataLoader(dataset, args.batchsize, num_workers=1)

itr = args.loaditerations

for item in dataloader:
    itr += 1
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
    # Test decoder :
    decoder.viterbiGenerate(ctx[:,-1], usrs_b[:,-1], word_emb, dataset)

    # Training:
    prob, log_prob = decoder(ctx[:,:-1:], wds_b[:,1:,:max_output_words],
                             usrs_b[:,1:], sentence_lengths_padded[:,1:], words_flat)
    loss = -log_prob
    opt.zero_grad()
    loss.backward()
    grad_norm = clip_grad(params, args.gradclip)
    loss, grad_norm = tonumpy(loss, grad_norm)
    loss = loss[0]
    print(loss)
    train_writer.add_summary(
            TF.Summary(
                value=[
                    TF.Summary.Value(tag='loss', simple_value=loss),
                    TF.Summary.Value(tag='grad_norm', simple_value=grad_norm),
                    ]
                ),
            itr
            )
    opt.step()
    if itr % 100 == 0:
        prob = decoder(ctx[:,:-1:], wds_b[:,1:,:max_output_words],
                             usrs_b[:,1:], sentence_lengths_padded[:,1:]).squeeze()
        #Energy defined as H here:https://en.wikipedia.org/wiki/Entropy_(information_theory)
        Energy = (prob.exp() * prob * -1).sum(1)
        E_mean, E_std, E_max, E_min = tonumpy(
                Energy.mean(), Energy.std(), Energy.max(), Energy.min())
        E_mean, E_std, E_max, E_min = [e[0] for e in [E_mean, E_std, E_max, E_min]]
        train_writer.add_summary(
            TF.Summary(
                value=[
                    TF.Summary.Value(tag='Energy_mean', simple_value=E_mean),
                    TF.Summary.Value(tag='Energy_std', simple_value=E_std),
                    TF.Summary.Value(tag='Energy_max', simple_value=E_max),
                    TF.Summary.Value(tag='Energy_min', simple_value=E_min),
                    ]
                ),
            itr
            )
        T.save(user_emb, '%s-user_emb-%05d' % (modelnamesave, itr))
        T.save(word_emb, '%s-word_emb-%05d' % (modelnamesave, itr))
        T.save(enc, '%s-enc-%05d' % (modelnamesave, itr))
        T.save(context, '%s-context-%05d' % (modelnamesave, itr))
        T.save(decoder, '%s-decoder-%05d' % (modelnamesave, itr))
    print('G', itr, tonumpy(loss))

    
    
    
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
