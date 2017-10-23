
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

from adv import *
from test import test


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

    def zero_state(self, batch_size):
        lstm_h = tovar(T.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size))
        lstm_c = tovar(T.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size))
        initial_state = (lstm_h, lstm_c)
        return initial_state

    def forward(self, sent_encodings, length, initial_state=None):
        num_layers = self._num_layers
        batch_size = sent_encodings.size()[0]
        context_size = self._context_size
        
        if initial_state is None:
            initial_state = self.zero_state(batch_size)
        sent_encodings = sent_encodings.permute(1,0,2)
        embed, (h, c) = dynamic_rnn(self.rnn, sent_encodings, length, initial_state)
        embed = embed.contiguous().view(-1, context_size)
        #h = h.permute(1, 0, 2)
        return embed.view(batch_size, -1, context_size), (h, c)

class Decoder(NN.Module):
    def __init__(self,size_usr, size_wd, context_size, num_words, 
                 state_size = None, num_layers=1):
        NN.Module.__init__(self)
        self._num_words = num_words
        self._context_size = context_size
        self._size_wd = size_wd
        self._size_usr = size_usr
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
        num_layers = self._num_layers
        batch_size = wd_emb.size()[0]
        maxlenbatch = wd_emb.size()[1]
        maxwordsmessage = wd_emb.size()[2]
        context_size = self._context_size
        size_wd = self._size_wd
        size_usr = self._size_usr
        state_size = self._state_size

        if initial_state is None:
            initial_state = self.zero_state(batch_size)
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
            return out, (h, c)#.contiguous().view(batch_size, maxlenbatch, maxwordsmessage, -1)
        else:
            target = T.cat((wd_target[:, :, 1:], tovar(T.zeros(batch_size, maxlenbatch, 1)).long()), 2)
            out = self.softmax(embed.view(-1, state_size), target.view(-1))
            out = out.view(batch_size, maxlenbatch, maxwordsmessage)
            mask = (target != 0).float()
            out = out * mask
            log_prob = out.sum() / mask.sum()
            return out, log_prob, (h, c)


parser = argparse.ArgumentParser(description='Ubuntu Dialogue dataset parser')
parser.add_argument('--dataroot', type=str,default='ubuntu', help='Root of the data downloaded from github')
parser.add_argument('--outputdir', type=str, default ='outputs',help='output directory')
parser.add_argument('--logdir', type=str, default='logs', help='log directory')
parser.add_argument('--encoder_layers', type=int, default=2)
parser.add_argument('--decoder_layers', type=int, default=2)
parser.add_argument('--context_layers', type=int, default=2)
parser.add_argument('--size_context', type=int, default=64)
parser.add_argument('--size_sentence', type=int, default=64)
parser.add_argument('--decoder_size_sentence', type=int, default=64)
parser.add_argument('--size_usr', type=int, default=16)
parser.add_argument('--size_wd', type=int, default=16)
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--gradclip', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--modelname', type=str, default = '')
parser.add_argument('--modelnamesave', type=str, default='')
parser.add_argument('--modelnameload', type=str, default='')
parser.add_argument('--loaditerations', type=int, default=0)
parser.add_argument('--max_sentence_length_allowed', type=int, default=100)
parser.add_argument('--max_turns_allowed', type=int, default=3)
parser.add_argument('--num_loader_workers', type=int, default=4)
parser.add_argument('--adversarial_sample', type=int, default=1)
args = parser.parse_args()

dataset = UbuntuDialogDataset(args.dataroot,
                              max_sentence_length_allowed=args.max_sentence_length_allowed,
                              max_turns_allowed=args.max_turns_allowed)
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
enc = cuda(Encoder(size_usr, size_wd, size_sentence, num_layers = args.encoder_layers))
context = cuda(Context(size_sentence, size_context, num_layers = args.context_layers))
decoder = cuda(Decoder(size_usr, size_wd, size_context, num_words+1,
                       decoder_size_sentence, num_layers = args.decoder_layers))

params = sum([list(m.parameters()) for m in [user_emb, word_emb, enc, context, decoder]], [])
opt = T.optim.Adam(params, lr=args.lr)


dataloader = UbuntuDialogDataLoader(dataset, args.batchsize, num_workers=args.num_loader_workers)

itr = args.loaditerations
epoch = 0

if modelnameload:
    if len(modelnameload) > 0:
        user_emb = T.load('%s-user_emb-%05d' % (modelnameload, args.loaditerations))
        word_emb = T.load('%s-word_emb-%05d' % (modelnameload, args.loaditerations))
        enc = T.load('%s-enc-%05d' % (modelnameload, args.loaditerations))
        context = T.load('%s-context-%05d' % (modelnameload, args.loaditerations))
        decoder = T.load('%s-decoder-%05d' % (modelnameload, args.loaditerations))

while True:
    epoch += 1
    for item in dataloader:
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
            wds_adv, usrs_adv, loss_adv = adversarial_word_users(wds_b, usrs_b, turns,
               size_wd,batch_size,size_usr,
               sentence_lengths_padded, enc, 
               context,words_padded, decoder)
            wds_b = tovar((wds_b + tovar(wds_adv)).data)
            usrs_b = tovar((usrs_b + tovar(usrs_adv)).data)
        max_turns = turns.max()
        max_words = wds_b.size()[2]
        encodings = enc(wds_b.view(batch_size * max_turns, max_words, size_wd),
                usrs_b.view(batch_size * max_turns, size_usr), 
                sentence_lengths_padded.view(-1))
        if itr % 10 == 4 and args.adversarial_sample == 1:
            wds_adv, usrs_adv, enc_adv, loss_adv = adversarial_encodings_wds_usrs(encodings, batch_size, 
                    wds_b,usrs_b,max_turns, context, turns, 
                    sentence_lengths_padded, words_padded, decoder)
            wds_b = tovar((wds_b + tovar(wds_adv)).data)
            usrs_b = tovar((usrs_b + tovar(usrs_adv)).data)
            encodings = tovar((encodings + tovar(enc_adv)).data)
        encodings = encodings.view(batch_size, max_turns, -1)
        ctx, _ = context(encodings, turns)
        if itr % 10 == 7 and args.adversarial_sample == 1:
            wds_adv, usrs_adv, ctx_adv, loss_adv = adversarial_context_wds_usrs(ctx, sentence_lengths_padded,
                      wds_b,usrs_b,words_padded, decoder)
            wds_b = tovar((wds_b + tovar(wds_adv)).data)
            usrs_b = tovar((usrs_b + tovar(usrs_adv)).data)
            ctx = tovar((ctx + tovar(ctx_adv)).data)
        max_output_words = sentence_lengths_padded[:, 1:].max()
        words_flat = words_padded[:,1:,:max_output_words].contiguous()
        # Training:
        prob, log_prob, _ = decoder(ctx[:,:-1:], wds_b[:,1:,:max_output_words],
                                 usrs_b[:,1:], sentence_lengths_padded[:,1:], words_flat)
        loss = -log_prob
        opt.zero_grad()
        loss.backward()
        grad_norm = clip_grad(params, args.gradclip)
        loss, grad_norm = tonumpy(loss, grad_norm)
        loss = loss[0]
        print(loss)
        opt.step()
        if itr % 10 == 1 and args.adversarial_sample == 1:
            train_writer.add_summary(
                TF.Summary(value=[TF.Summary.Value(tag='wd_usr_adv_diff', simple_value=loss_adv - loss)]),itr)
        if itr % 10 == 4 and args.adversarial_sample == 1:
            train_writer.add_summary(
                TF.Summary(value=[TF.Summary.Value(tag='enc_adv_diff', simple_value=loss_adv - loss)]),itr)
        if itr % 10 == 7 and args.adversarial_sample == 1:
            train_writer.add_summary(
                TF.Summary(value=[TF.Summary.Value(tag='ctx_adv_diff', simple_value=loss_adv - loss)]),itr)
        mask = mask_4d(wds_b.size(), turns , sentence_lengths_padded)
        wds_dist = wds_b* mask
        mask = mask_3d(usrs_b.size(), turns)
        usrs_dist = usrs_b * mask
        mask = mask_3d(encodings.size(), turns)
        sent_dist = encodings * mask
        ctx_dist = ctx * mask
        wds_dist, usrs_dist, sent_dist, ctx_dist = tonumpy(wds_dist, usrs_dist, sent_dist, ctx_dist)
        if itr % 10 == 9:
            train_writer.add_summary(
                    TF.Summary(
                        value=[
                            TF.Summary.Value(tag='loss', simple_value=loss),
                            TF.Summary.Value(tag='grad_norm', simple_value=grad_norm),
                            TF.Summary.Value(tag='wd_std', simple_value=np.nanstd(wds_dist)),
                            TF.Summary.Value(tag='usr_std', simple_value=np.nanstd(usrs_dist)),
                            TF.Summary.Value(tag='sent_std', simple_value=np.nanstd(sent_dist)),
                            TF.Summary.Value(tag='ctx_std', simple_value=np.nanstd(ctx_dist)),
                            ]
                        ),
                    itr
                    )

        # Beam search test
        words = tonumpy(words_padded.data[0, ::2])
        full_turns = words.shape[0]
        sentence_lengths = tonumpy(sentence_lengths_padded.data[0, ::2])
        initiator = speaker_padded.data[0, 0]
        respondent = speaker_padded.data[0, 1]
        words_nopad = [list(words[i, :sentence_lengths[i]]) for i in range(full_turns)]
        dialogue, scores = test(dataset, enc, context, dec, word_emb, user_emb, words_nopad,
                                initiator, respondent, args.max_sentence_length)
        dialogue_strings = dataset.translate_item(None, None, dialogue)
        for d, ds, s in zip(dialogue, dialogue_strings, scores):
            print(d, ds, s)

        if itr % 100 == 0:
            prob, _ = decoder(ctx[:4,:-1], wds_b[:4,1:,:max_output_words],
                                 usrs_b[:4,1:], sentence_lengths_padded[:4,1:]).squeeze()
            #Entropy defined as H here:https://en.wikipedia.org/wiki/Entropy_(information_theory)
            mask = mask_4d(prob.size(), turns[:4] -1 , sentence_lengths_padded[:4,1:])
            Entropy = (prob.exp() * prob * -1) * mask
            Entropy_per_word = Entropy.sum(-1)
            Entropy_per_word = tonumpy(Entropy_per_word)[0]
            #E_mean = tonumpy(Entropy_per_word.sum() / mask.sum())[0]
            #E_mean, E_std, E_max, E_min = tonumpy(
            #        Entropy_per_word.mean(), Entropy.std(), Entropy.max(), Entropy.min())
            
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
            T.save(user_emb, '%s-user_emb-%05d' % (modelnamesave, itr))
            T.save(word_emb, '%s-word_emb-%05d' % (modelnamesave, itr))
            T.save(enc, '%s-enc-%05d' % (modelnamesave, itr))
            T.save(context, '%s-context-%05d' % (modelnamesave, itr))
            T.save(decoder, '%s-decoder-%05d' % (modelnamesave, itr))
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
