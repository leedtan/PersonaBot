
from data_loader_stage1 import *
from modules import *
import torch as T
import torch.nn as NN
import torch.nn.functional as F
from encoder import Encoder

dataroot = '/misc/vlgscratch4/ChoGroup/gq/data/OpenSubtitles/OpenSubtitles-dialogs-small'
metaroot = 'opensub'
max_sentence_length_allowed = 20
max_turns_allowed = 6
vocabsize = 159996
size_usr = 10
size_wd = 50

dataset = UbuntuDialogDataset(os.path.join(dataroot, 'SouthPark'),
                              wordcount_pkl=metaroot + '/wordcount.pkl',
                              usercount_pkl=metaroot + '/usercount.pkl',
                              turncount_pkl=metaroot + '/turncount.pkl',
                              max_sentence_lengths_pkl=metaroot + '/max_sentence_lengths.pkl',
                              max_sentence_length_allowed=max_sentence_length_allowed,
                              max_turns_allowed=max_turns_allowed,
                              vocab_size=vocabsize)
dataloader = UbuntuDialogDataLoader(dataset, 4, num_workers=0)
it = iter(dataloader)

nusers = len(dataset.users)
nvocab = len(dataset.vocab)
user_embedder = NN.Embedding(nusers+1, size_usr, padding_idx = 0, scale_grad_by_freq=True)
word_embedder = NN.Embedding(nvocab+1, size_wd, padding_idx = 0, scale_grad_by_freq=True)
encoder = Encoder(size_usr, size_wd, 100, 2)

turns, sentence_lengths_padded, speaker_padded, \
    addressee_padded, words_padded, words_reverse_padded = tovar(*next(it))
speaker_padded = speaker_padded.long()
words_padded = words_padded.long()
turns = turns.long()
sentence_lengths_padded = sentence_lengths_padded.long()

speaker_padded_size = speaker_padded.size()
words_padded_size = words_padded.size()
user_emb = user_embedder(speaker_padded.view(-1)).view(*speaker_padded_size, size_usr)
word_emb = word_embedder(words_padded.view(-1)).view(*words_padded_size, size_wd)
import ipdb
ipdb.set_trace()
output, anno = encoder(word_emb, user_emb, turns, sentence_lengths_padded)
print(output)
print(anno)
