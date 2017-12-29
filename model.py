
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
def lrelu(x):
    return tf.maximum(x, .1*x)
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

def encode_sentence(x, layers, size, lengths, cells):
    prev_layer = x
    for idx in range(layers):
        h,c = nn.bidirectional_dynamic_rnn(
            cell_fw=cells[idx][0], cell_bw=cells[idx][1],
            inputs=prev_layer,
            sequence_length=lengths,
            initial_state_fw=None,
            initial_state_bw=None,
            dtype=tf.float32,
            parallel_iterations=None,
            swap_memory=False,
            time_major=False,
            scope='enc' + str(idx)
            )
        prev_layer = tf.concat(h, -1)
    output_layer = prev_layer[:,-1,:]
    return output_layer

def encode_context(x, layers, size, lengths, cells):
    prev_layer = x
    for idx in range(layers):
        prev_layer,c = nn.dynamic_rnn(
            cell=cells[idx],
            inputs=prev_layer,
            sequence_length=lengths,
            initial_state=None,
            dtype=tf.float32,
            parallel_iterations=None,
            swap_memory=False,
            time_major=False,
            scope='ctx' + str(idx)
            )
    return prev_layer

def decode(x, layers, size, lengths, cells):
    prev_layer = x
    for idx in range(layers):
        prev_layer,c = nn.dynamic_rnn(
            cell=cells[idx],
            inputs=prev_layer,
            sequence_length=lengths,
            initial_state=None,
            dtype=tf.float32,
            parallel_iterations=None,
            swap_memory=False,
            time_major=False,
            scope='dec' + str(idx)
            )
    return prev_layer

class Model():
    def __init__(self, layers_enc=1, layers_ctx=1, layers_dec=1,
                 size_usr = 32, size_wd = 32,
                 size_enc=32, size_ctx=32, size_dec=32,
                 num_wds = 8192,num_usrs = 1000):
        self.layers_enc = layers_enc
        self.layers_ctx = layers_ctx
        self.layers_dec = layers_dec
        self.size_usr = size_usr
        self.size_wd = size_wd
        self.size_enc = size_enc
        self.size_ctx = size_ctx
        self.size_dec = size_dec
        self.num_wds = num_wds
        self.num_srs = num_usrs
        
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
        
        self.mask = tf.sequence_mask(self.sentence_lengths_flat)
        '''
        self.mask = tf.stack([tf.concat((tf.ones(sent_len), tf.zeros(self.max_sentence_length - sent_len))) 
            for sent_len in self.sentence_lengths_flat], 0)
    
        '''
        self.mask_flat = tf.cast(tf.reshape(self.mask, [-1]), tf.float32)
        self.mask_expanded = tf.reshape(self.mask, [self.batch_size, self.max_conv_len, self.max_sent_len])
        self.mask_decode = self.mask_expanded[:,1:,:]
        self.mask_flat_decode = tf.cast(tf.reshape(self.mask_decode, [-1]), tf.float32)
        
        self.wd_mat = tf.Variable(1e-3*tf.random_normal([num_wds, size_wd]))
        self.usr_mat = tf.Variable(1e-3*tf.random_normal([num_usrs, size_usr]))
        
        self.wd_emb = tf.nn.embedding_lookup(self.wd_mat, self.wd_ind)
        self.usr_emb = tf.nn.embedding_lookup(self.usr_mat, self.usr_ind)
        self.usr_emb_expanded = tf.tile(tf.expand_dims(self.usr_emb, 2),[1,1,self.max_sent_len,1])
        
        self.wds_usrs = tf.concat((self.wd_emb, self.usr_emb_expanded), 3)
        
        self.sent_rnns = [[GRUCell(size_enc//2, kernel_initializer=tf.contrib.layers.xavier_initializer()),
                           GRUCell(size_enc//2, kernel_initializer=tf.contrib.layers.xavier_initializer())]
            for _ in range(layers_enc)]
        self.ctx_rnns = [GRUCell(size_ctx, kernel_initializer=tf.contrib.layers.xavier_initializer())
            for _ in range(layers_ctx)]
        self.dec_rnns = [GRUCell(size_dec, kernel_initializer=tf.contrib.layers.xavier_initializer())
            for _ in range(layers_dec)]
        
        self.sent_input = tf.reshape(self.wds_usrs,[-1, self.max_sent_len, self.size_wd + self.size_usr])
        #        self.batch_size, self.max_conv_len, self.max_sent_len, self.size_wd + self.size_usr])
        
        self.sentence_encs = encode_sentence(
                x=self.sent_input, layers=layers_enc, size = size_enc, 
                lengths=self.sentence_lengths_flat, cells = self.sent_rnns)
        
        sentence_encs = tf.reshape(self.sentence_encs, [self.batch_size, self.max_conv_len,self.size_enc])
        
        self.context_encs = encode_context(
                x = sentence_encs, layers = layers_ctx, size = size_ctx,
                lengths = self.conv_lengths_flat, cells = self.ctx_rnns)
        
        #batch, messages, (expansion needed), emb_dim
        
        dec_inputs = tf.concat(
                (tf.tile(tf.expand_dims(self.context_encs, 2),[1,1,self.max_sent_len,1]), self.wds_usrs), 3)
        
        self.dec_inputs = tf.reshape(dec_inputs, [-1, self.max_sent_len,self.size_ctx + self.size_wd + self.size_usr])
        
        self.target = self.wd_ind[:,1:, :]
        self.target_flat = tf.reshape(self.target, [-1])
        
        self.decoder_out = decode(
                x = self.dec_inputs, layers=layers_dec, size=size_dec,
                lengths = self.sentence_lengths_flat, cells = self.dec_rnns)
        #decoder out is message0: message-1
        self.decoder_deep = tf.reshape(
                self.decoder_out, (self.batch_size, self.max_conv_len, self.max_sent_len, self.size_dec))
        self.decoder_out_for_ppl = tf.reshape(self.decoder_deep[:,:-1,:,:],(-1, size_dec))
        self.dec_raw = layers.dense(self.decoder_out_for_ppl, self.num_wds)
        self.dec_relu = lrelu(self.dec_raw)
        
        '''
        tf.nn.sampled_softmax_loss(
            weights,
            biases,
            labels,
            inputs,
            num_sampled,
            num_classes,
            num_true=1,
            sampled_values=None,
            remove_accidental_hits=True,
            partition_strategy='mod',
            name='sampled_softmax_loss'
        )
        if mode == "train":
          loss = tf.nn.sampled_softmax_loss(
          weights=weights,
          biases=biases,
          labels=labels,
          inputs=inputs,
          ...,
          partition_strategy="div")
        elif mode == "eval":
          logits = tf.matmul(inputs, tf.transpose(weights))
          logits = tf.nn.bias_add(logits, biases)
          labels_one_hot = tf.one_hot(labels, n_classes)
          loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labels_one_hot,
          logits=logits)
        self.softmax_weights = tf.get_variable("softmax_weights", [self.dec_relu.get_shape(1), num_wds], tf.float32,
                                 tf.contrib.layers.xavier_initializer())
        self.softmax_bias = tf.get_variable("softmax_bias", [num_wds],
            initializer=tf.constant_initializer(0))
        self.ppl_loss_raw = tf.nn.sampled_softmax_loss(
            weights=self.softmax_weights,
            biases = self.softmax_bias,
            labels = self.labels,
            inputs = self.dec_relu,
            num_sampled,
            num_classes,
            num_true=1,
            sampled_values=None,
            remove_accidental_hits=True,
            partition_strategy='mod',
            name='sampled_softmax_loss'
        )
        #self.ppl_loss_raw = tf.gather_nd(tf.transpose(self.dec_softmaxed), labels)[:,0]
        #self.ppl_loss_raw = tf.gather_nd(self.dec_softmaxed,self.target_flat)
        self.ppl_loss_raw = self.dec_softmaxed[self.target_flat]
        self.ppl_loss_raw = self.softmax(self.decoder_out_for_ppl, self.target)
        self.dec_softmaxed = tf.nn.softmax(self.dec_relu)
        '''
        '''
        OTHER VERSION:
        self.softmax_weights = tf.get_variable("softmax_weights2", [self.dec_relu.get_shape()[1], num_wds], tf.float32,
                                 tf.contrib.layers.xavier_initializer())
        self.softmax_bias = tf.get_variable("softmax_bias2", [num_wds],
            initializer=tf.constant_initializer(0))
        self.labels = tf.one_hot(indices = self.target_flat, depth = self.num_wds)
        self.ppl_loss_raw = tf.nn.sampled_softmax_loss(
            weights=self.softmax_weights,
            biases = self.softmax_bias,
            labels = self.labels,
            inputs = self.dec_relu,
            num_sampled = num_wds,
            num_classes = num_wds,
            num_true=1,
            sampled_values=None,
            remove_accidental_hits=True,
            partition_strategy='mod',
            name='sampled_softmax_loss'
        )
        '''
        
        self.labels = tf.one_hot(indices = self.target_flat, depth = self.num_wds)
        self.ppl_loss_raw = tf.nn.softmax_cross_entropy_with_logits(
                labels = self.labels, logits = self.dec_relu)
        self.ppl_loss_masked =  self.ppl_loss_raw * self.mask_flat_decode
        self.ppl_loss = tf.reduce_sum(self.ppl_loss_masked)/tf.reduce_sum(self.mask_flat_decode)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        
        gvs = optimizer.compute_gradients(self.ppl_loss * 1000)
        self.grad_norm = tf.reduce_mean([tf.reduce_mean(tf.square(grad)) for grad, var in gvs if grad is not None])
        clip_norm = 100
        clip_single = 1
        capped_gvs = [(tf.clip_by_value(grad, -1*clip_single,clip_single), var)
                      for grad, var in gvs if grad is not None]
        capped_gvs = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in capped_gvs if grad is not None]
        self.optimizer = optimizer.apply_gradients(capped_gvs)
        
        
        
 
parser = argparse.ArgumentParser(description='Ubuntu Dialogue dataset parser')
parser.add_argument('--dataroot', type=str,default='OpenSubtitles-dialogs-small', help='Root of the data downloaded from github')
parser.add_argument('--metaroot', type=str, default='opensub', help='Root of meta data')
#796
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
parser.add_argument('--size_usr', type=int, default=16)
parser.add_argument('--size_wd', type=int, default=50)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--gradclip', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--modelname', type=str, default = '')
parser.add_argument('--modelnamesave', type=str, default='')
parser.add_argument('--modelnameload', type=str, default='')
parser.add_argument('--loaditerations', type=int, default=0)
parser.add_argument('--max_sentence_length_allowed', type=int, default=30)
parser.add_argument('--max_turns_allowed', type=int, default=12)
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

train_writer = tf.summary.FileWriter(log_train)

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



'''
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
'''
model = Model(layers_enc=1, layers_ctx=1, layers_dec=1,
                 size_usr = 32, size_wd = 32,
                 size_enc=32, size_ctx=32, size_dec=32,
                 num_wds = num_words+1,num_usrs = num_usrs+1)



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
'''
params = sum([list(m.parameters()) for m in [user_emb, word_emb, enc, context, decoder]], [])
named_params = sum([list(m.named_parameters())
    for m in [user_emb, word_emb, enc, context, decoder]], [])

def enable_train(sub_modules):
    for m in sub_modules:
        m.train()
def enable_eval(sub_modules):
    for m in sub_modules:
        m.eval()
opt = T.optim.Adam(params, lr=args.lr,weight_decay=1e-8)
'''
#opt = T.optim.RMSprop(params, lr=args.lr,weight_decay=1e-6)
adv_style = 0
scatter_entropy_freq = 200
time_train = 0
time_decode = 0
while True:
    epoch += 1
    for item in dataloader:
        start_train = time.time()
        if itr % scatter_entropy_freq == 0:
            adv_style = 1 - adv_style
            #adjust_learning_rate(opt, args.lr / np.sqrt(1 + itr / 10000))
        itr += 1
        '''
        if itr % 100 == 0:
            for p, v in named_params:
                print(p, v.data.min(), v.data.max())
        '''
        turns, sentence_lengths_padded, speaker_padded, \
            addressee_padded, words_padded, words_reverse_padded = item
        if args.plot_attention:
            list_stringify_sentence = batch_to_string_sentence(words_padded, sentence_lengths_padded, dataset)
        else:
            list_stringify_sentence = None

        wds = sentence_lengths_padded.sum()
        max_wds = args.max_turns_allowed * args.max_sentence_length_allowed
        if sentence_lengths_padded.shape[1] < 2:
            continue
        if wds > max_wds * .6 or wds < max_wds * .05:
            continue
        
        
        batch_size = turns.shape[0]
        max_turns = words_padded.shape[1]
        max_words = words_padded.shape[2]
        #batch, turns in a sample, words in a message
        feed_dict = {
                model.learning_rate: 1e-3,
                model.wd_ind: words_padded,
                model.usr_ind: speaker_padded,
                model.conv_lengths: turns,
                model.sentence_lengths: sentence_lengths_padded,
                model.batch_size: batch_size,
                model.max_sent_len: max_words,
                model.max_conv_len: max_turns
                }
        '''
         self.dec_softmaxed = tf.nn.softmax(self.dec_relu)
        self.ppl_loss_raw = tf.gather_nd(self.dec_softmaxed, tf.reshape(self.target_flat, [-1,1]))[:,0]
        self.ppl_loss_masked =  self.ppl_loss_raw * self.mask_flat
        self.ppl_loss = tf.reduce_mean(self.ppl_loss_masked)
        '''
        #print(words_padded.shape)
        #_, loss, grad_norm =  sess.run(model.dec_softmaxed,feed_dict=feed_dict)
        dbg = 0
        if dbg:
            self=model
            feed_dict_dbg = {self.dec_relu : np.stack([np.arange(40000),np.arange(40000-1,-1,-1)]),
                             self.target_flat : np.array([39999,0], dtype = np.int32)}

            feed_dict_1 = {self.dec_softmaxed : np.stack([np.arange(40000),np.arange(40000-1,-1,-1)])}
            feed_dict_2 = {self.target_flat : np.array([0,5], dtype = np.int32)}
            #sess.run([self.dec_softmaxed,self.target_flat], feed_dict_dbg)
            #sess.run([self.dec_softmaxed], feed_dict_1)
            #sess.run([self.target_flat], feed_dict_2)
            sess.run(tf.nn.softmax_cross_entropy_with_logits(
                labels = self.labels, logits = self.dec_relu), feed_dict_dbg)
            sess.run([self.labels, self.dec_relu], feed_dict_dbg)
            '''
        self.labels = tf.one_hot(indices = self.target_flat, depth = self.num_wds)
        self.ppl_loss_raw = tf.nn.softmax_cross_entropy_with_logits(
                labels = self.labels, logits = self.dec_relu)
        self.ppl_loss_masked =  self.ppl_loss_raw * self.mask_flat_decode
        self.ppl_loss = tf.reduce_sum(self.ppl_loss_masked)/tf.reduce_sum(self.mask_flat_decode)
        
            tflat, soft = sess.run(tf.nn.softmax_cross_entropy_with_logits(
                labels = tf.reshape(self.target_flat, [1,-1]), logits = tf.transpose(self.dec_softmaxed)),feed_dict)
            tflat, soft = sess.run(tf.nn.softmax_cross_entropy_with_logits(
                labels = self.target_flat, logits = tf.transpose(self.dec_softmaxed)),feed_dict)
            tflat, soft = sess.run(tf.nn.softmax_cross_entropy_with_logits(
                labels = tf.reshape(self.target_flat, [1, -1]), logits = self.dec_softmaxed),feed_dict)
            
            tflat, soft = sess.run([self.target_flat, self.dec_softmaxed],feed_dict)
            sess.run(tf.gather_nd(tf.transpose(self.dec_softmaxed), tf.reshape(self.target_flat, [-1,1]))[:,0], feed_dict_dbg)
            sess.run(tf.gather_nd(tf.transpose(self.dec_softmaxed), tf.reshape(self.target_flat, [-1,1]))[:,0], feed_dict)
            '''
            
            _, loss, grad_norm, dec_relu, loss_raw, dec_masked, labels =  sess.run(
                [model.optimizer,model.ppl_loss, model.grad_norm,
                model.dec_relu, model.ppl_loss_raw,
                model.ppl_loss_masked, model.labels],
                feed_dict=feed_dict)
            
        else:
            _, loss, grad_norm =  sess.run([model.optimizer,model.ppl_loss, model.grad_norm],
                    feed_dict=feed_dict)
        
        
        
        if itr % 10 == 0:
            train_writer.add_summary(
                    tf.Summary(
                        value=[
                            tf.Summary.Value(tag='perplexity', simple_value=np.exp(np.min((8, tonumpy(loss))))),
                            tf.Summary.Value(tag='loss', simple_value=loss),
                            tf.Summary.Value(tag='grad_norm', simple_value=grad_norm),
                            
                            ]
                        ),
                    itr
                    )
        if itr % 100 == 0:
            a = 2
        time_train += time.time() - start_train
        #TODO 
        if itr % scatter_entropy_freq == 0 and 0:
            prob =  sess.run(model.dec_softmaxed,
                    feed_dict=feed_dict)
            #prob, _ = decoder(ctx[:1,:-1], wds_first_sentence_removed[:1,:,:].contiguous(),
            #                     usrs_decode[:1:], sentence_lengths_padded[:1,1:])
            #Entropy defined as H here:https://en.wikipedia.org/wiki/Entropy_(information_theory)
            
            #mask = 
            #mask = mask_4d(prob.size(), turns[:1] -1 , sentence_lengths_padded[:1,1:])
            prob = prob.reshape(batch_size, max_turns-1, max_words,-1)
            prob[:,turns[:1]:,:,:] = 0
            Entropy = (np.exp(prob) * prob * -1) * mask
            Entropy_per_word = Entropy.sum(-1)
            Entropy_per_word = tonumpy(Entropy_per_word)[0]
            #E_mean = tonumpy(Entropy_per_word.sum() / mask.sum())[0]
            #E_mean, E_std, E_max, E_min = tonumpy(
            #        Entropy_per_word.mean(), Entropy.std(), Entropy.max(), Entropy.min())
            E_mean = np.nanmean(Entropy_per_word)
            train_writer.add_summary(
                tf.Summary(
                    value=[
                        tf.Summary.Value(tag='Entropy_mean', simple_value=E_mean)
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
                tf.Summary(
                    value=[
                        tf.Summary.Value(tag='Entropy_mean', simple_value=E_mean),
                        tf.Summary.Value(tag='Entropy_std', simple_value=E_std),
                        tf.Summary.Value(tag='Entropy_max', simple_value=E_max),
                        tf.Summary.Value(tag='Entropy_min', simple_value=E_min),
                        ]
                    ),
                itr
                )
            '''
            if args.adversarial_sample == 1 and itr > 1000:
                '''add_scatterplot(train_writer, losses=[adv_emb_diffs, adv_sent_diffs, adv_ctx_diffs], 
                                scales=[adv_emb_scales, adv_sent_scales, adv_ctx_scales], 
                                names=['embeddings', 'sentence', 'context'], itr = itr, 
                                log_dir = log_train, tag = 'scatterplot', style=adv_style)
                '''
                adv_emb_diffs = []
                adv_sent_diffs = []
                adv_ctx_diffs = []
                adv_emb_scales = []
                adv_sent_scales = []
                adv_ctx_scales = []
        # ...Tensorboard viz end

        # Train with Policy Gradient on BLEU scores once for a while.
        if itr % 10 == 0 and itr > 100 and 0:
            start_decode = time.time()
            #enable_eval([user_emb, word_emb, enc, context, decoder])
            greedy_responses, logprobs = decoder.greedyGenerateBleu(
                    ctx[:1,:-1,:].view(-1, size_context + size_attn),
                      usrs_decode[:1,:,:].view(-1, size_usr), word_emb, dataset)
            # Only take the first turns[0] responses
            greedy_responses = greedy_responses[:turns[0]]
            logprobs = logprobs[:turns[0]]
            reference = tonumpy(words_padded[0,1:turns[0],:])
            hypothesis = tonumpy(greedy_responses)
            logprobs_np = tonumpy(logprobs)
            
            # Compute BLEU scores
            real_sent = []
            gen_sent = []
            BLEUscores = []
            BLEUscoresplot = []
            lengths_gen = []
            batch_words = Counter()
            batch_bigrams = Counter()
            batch_trigrams = Counter()
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
                batch_bigrams.update([tuple(gen_sent[-1][i:i+2]) for i in range(len(gen_sent[-1]))])
                batch_trigrams.update([tuple(gen_sent[-1][i:i+3]) for i in range(len(gen_sent[-1])-1)])
                curr_bleu = bleu_score.sentence_bleu(
                        [real_sent[-1]], gen_sent[-1], smoothing_function=smoother.method1)
                BLEUscoresplot.append(curr_bleu)
                curr_bleu += num_words / (1+np.sqrt(itr))
                
                #curr_bleu += extra_penalty[num_words]/(1+np.sqrt(itr))
                BLEUscores.append(curr_bleu)
            
            # Use BLEU scores as reward, comparing it to baseline (moving average)
            baseline = np.mean(BLEUscores) if baseline is None else baseline * 0.5 + np.mean(BLEUscores) * 0.5
            reward = np.array(BLEUscores) - baseline
            reward = reward.reshape(-1, 1).repeat(logprobs_np.shape[1], axis=1)
            total_words = sum(batch_words.values())
            total_bigrams = max([10, sum(batch_bigrams.values())])
            total_trigrams = max([10, sum(batch_trigrams.values())])
            tot_unigram_penalty = 0
            tot_bigram_penalty = 0
            tot_trigram_penalty = 0
            for sentence_idx in range(hypothesis.shape[0]):
                for word_idx in range(1,lengths_gen[sentence_idx]):
                    unigram_count = batch_words[hypothesis[sentence_idx,word_idx]]
                    if unigram_count > 6:
                        unigram_penalty = (unigram_count/total_words)**2 * 1
                        reward[sentence_idx,word_idx-1] -= unigram_penalty * args.lambda_repetitive
                        tot_unigram_penalty += unigram_penalty * args.lambda_repetitive
            
            for sentence_idx in range(hypothesis.shape[0]):
                for word_idx in range(0,lengths_gen[sentence_idx]):
                    bigram_count = batch_bigrams[tuple(hypothesis[sentence_idx,word_idx:word_idx+2])]
                    if bigram_count > 4:
                        #.1 is transition. yields .02, and .02
                        bigram_penalty = (bigram_count / total_bigrams) * .1 + \
                            ((bigram_count / total_bigrams) ** 2) * 1
                        min_c = max([word_idx-1,0])
                        for ci in range(min_c, word_idx+1):
                            if ci >= reward.shape[1]:
                                continue
                            reward[sentence_idx,ci] -= bigram_penalty * args.lambda_repetitive
                            tot_bigram_penalty += bigram_penalty * args.lambda_repetitive
            
            for sentence_idx in range(hypothesis.shape[0]):
                for word_idx in range(0,lengths_gen[sentence_idx]-1):
                    trigram_count = batch_trigrams[tuple(hypothesis[sentence_idx,word_idx:word_idx+3])]
                    if trigram_count > 2:
                        #.1 is transition of loss importance. yields .03 from first loss, .03 from second loss
                        trigram_penalty = (trigram_count / total_trigrams) * 1 + \
                            ((trigram_count / total_trigrams) ** 2) * 10
                        min_c = max([word_idx-1,0])
                        #max_c = min([word_idx+1, reward.shape[1]])
                        for ci in range(min_c, word_idx + 2):
                            if ci >= reward.shape[1]:
                                continue
                            reward[sentence_idx,ci] -= trigram_penalty * args.lambda_repetitive
                            tot_trigram_penalty += trigram_penalty * args.lambda_repetitive
                        
            if not np.all(~np.isnan(tonumpy(reward))):
                print('crash 5')
                continue
            #assert np.all(~np.isnan(reward))
            for idx in range(reference.shape[0]):
                if lengths_gen[idx] < reward.shape[1]:
                    reward[idx,lengths_gen[idx]:] = 0
            
            # Set the head gradients of the log-probabilities as negative of reward
            opt.zero_grad()
            logprobs.backward(args.lambda_pg * -cuda(T.Tensor(reward.T)))
            pg_grads = {p: p.grad.data.clone() for p in params if p.grad is not None}
            pg_grad_norm = sum(T.norm(v) for v in pg_grads.values()) ** 0.5
            if itr % 10 == 0:
                print('Grad norm', grad_norm, 'PG Grad norm', pg_grad_norm)
            train_writer.add_summary(
                    tf.Summary(
                        value=[
                            tf.Summary.Value(tag='Average BLEU', simple_value=np.mean(BLEUscoresplot)),
                            tf.Summary.Value(tag='pg_grad_norm', simple_value=pg_grad_norm),
                            tf.Summary.Value(tag='unigram_penalty', simple_value=tot_unigram_penalty),
                            tf.Summary.Value(tag='bigram_penalty', simple_value=tot_bigram_penalty),
                            tf.Summary.Value(tag='trigram_penalty', simple_value=tot_trigram_penalty),
                            ]
                        ),
                    itr
                    )
            '''
            for idx in range(reference.shape[0]):
                greedy_responses[idx,:num_words].reinforce(BLEUscores[idx] - baseline)
            '''

            # Dump...
            #print('REAL:',dataset.translate_item(None, None, tonumpy(words_padded[:1,:,:])))
            if itr % 100 == -1:
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
                                print('Fake:', speaker[0], ' '.join(words[0]))
                                printed = 1
                    if printed == 0:
                        speaker, _, words = dataset.translate_item(tonumpy(speaker_padded[0:1, i+1]), None, greedy_responses[i:i+1,:])
                        print('Fake:', speaker[0], ' '.join(words[0]))
                    if words_padded_decode[i, 1].sum() == 0:
                        break
            time_decode += time.time() - start_decode
        
        # After all these gibberish, add back the recorded grads from PPL and take a step
       
        if itr % 10000 == 0:
            T.save(user_emb, '%s-user_emb-%08d' % (modelnamesave, itr))
            T.save(word_emb, '%s-word_emb-%08d' % (modelnamesave, itr))
            T.save(enc, '%s-enc-%08d' % (modelnamesave, itr))
            T.save(context, '%s-context-%08d' % (modelnamesave, itr))
            T.save(decoder, '%s-decoder-%08d' % (modelnamesave, itr))
        if itr % 10 == 0:
            print('Epoch', epoch, 'Iteration', itr, 'Loss', tonumpy(loss), 'PPL', np.exp(np.min((10, tonumpy(loss)))))

    
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
