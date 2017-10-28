import argparse
import os
import h5py
import numpy as np
import nltk
import pickle
import copy
from collections import Counter, OrderedDict

parser = argparse.ArgumentParser(description='Ubuntu Dialogue dataset parser')
parser.add_argument('--dataroot', type=str, required=True, help='Root of the data downloaded from github')
parser.add_argument('--wordcount', type=str, required=True, help='Stage1 wordcount.pkl path')
parser.add_argument('--usercount', type=str, required=True, help='Stage1 usercount.pkl path')
parser.add_argument('--turncount', type=str, required=True, help='Stage1 turncount.pkl path')
parser.add_argument('--max_sentence_lengths', type=str, required=True, help='Stage1 max_sentence_lengths.pkl path')
parser.add_argument('--valid_size', type=float, default=5000, help='valid set size')
parser.add_argument('--test_size', type=float, default=5000, help='test set size')
parser.add_argument('--min_word_occurrence', type=int, default=1)
parser.add_argument('--min_user_occurrence', type=int, default=5)
args = parser.parse_args()

with open(args.wordcount, 'rb') as f:
    wordcount = pickle.load(f)
with open(args.usercount, 'rb') as f:
    usercount = pickle.load(f)
with open(args.turncount, 'rb') as f:
    turncount = pickle.load(f)
with open(args.max_sentence_lengths, 'rb') as f:
    max_sentence_lengths = pickle.load(f)

word_selected = [w for w, c in wordcount.items() if c > args.min_word_occurrence]
user_selected = [u for u, c in usercount.items() if c > args.min_user_occurrence]
print '%d words selected' % len(word_selected)
print '%d users selected' % len(user_selected)

# Index meanings:
# >=1: Valid
# |V| + 1: Unknown
# |V| + 2: EOS
# 0: Null/padding
word_index = OrderedDict((w, i + 1) for i, w in enumerate(word_selected))
user_index = OrderedDict((u, i + 1) for i, u in enumerate(user_selected))
user_unk = len(user_selected) + 1
word_unk = len(word_selected) + 1
word_eos = len(word_selected) + 2

print 'Creating dataset...'
h5 = h5py.File('dataset.h5', 'w')
for l in turncount:
    # Dataset format:
    # SPEAKER-ID ADDRESSEE-ID SENTENCE-LENGTH WORD1 WORD2 ... WORDN EOS 0 0 0
    h5.create_dataset(
            str(l),
            shape=(turncount[l], l, max_sentence_lengths[l] + 4),
            dtype=np.int32,
            compression='gzip'
            )

turns_index = {}
for dir_ in os.listdir(args.dataroot):
    subdir = os.path.join(args.dataroot, dir_)
    files = [f for f in os.listdir(subdir) if f.endswith('.pkl')]
    for pkl in files:
        path = os.path.join(subdir, pkl)
        print path
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        speaker_list = obj['speaker']
        addressee_list = obj['addressee']
        word_list = obj['words']

        l = len(speaker_list)
        buf = np.zeros((l, max_sentence_lengths[l] + 4), dtype=np.int32)
        for i, (speaker, addressee, words) in enumerate(zip(speaker_list, addressee_list, word_list)):
            buf[i, 0] = user_index.get(speaker, user_unk)
            buf[i, 1] = user_index.get(addressee, user_unk)
            buf[i, 2] = len(words) + 1 # EOS
            buf[i, 3:len(words) + 3] = [word_index.get(w, word_unk) for w in words]
            buf[i, len(words) + 3] = word_eos

        h5[str(l)][turns_index.get(l, 0)] = buf
        turns_index[l] = turns_index.get(l, 0) + 1

train_index = {k: list(range(turns_index[k])) for k in turns_index.keys()}
valid_index = {}
test_index = {}
for i in range(args.valid_size):
    k = RNG.choice(train_index.keys())
    if k not in valid_index:
        valid_index[k] = []
    idx = RNG.choice(train_index[k])
    valid_index[k].append(idx)
    train_index[k].remove(idx)
    if len(train_index[k]) == 0:
        del train_index[k]
for i in range(args.test_size):
    k = RNG.choice(train_index.keys())
    if k not in test_index:
        test_index[k] = []
    idx = RNG.choice(train_index[k])
    test_index[k].append(idx)
    train_index[k].remove(idx)
    if len(train_index[k]) == 0:
        del train_index[k]

for k in valid_index:
    for i in valid_index[k]:
        assert i not in train_index.get(k, [])
for k in test_index:
    for i in test_index[k]:
        assert i not in train_index.get(k, [])

with open('train_index.pkl', 'wb') as f:
    pickle.dump(train_index, f)
with open('valid_index.pkl', 'wb') as f:
    pickle.dump(valid_index, f)
with open('test_index.pkl', 'wb') as f:
    pickle.dump(test_index, f)
