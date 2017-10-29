import argparse
import torch
import os
import h5py
import numpy as np
import nltk
import pickle
from collections import Counter, OrderedDict

parser = argparse.ArgumentParser(description='Ubuntu Dialogue dataset parser')
parser.add_argument('--dataroot', type=str, required=True, help='Root of the data downloaded from github')
args = parser.parse_args()

print 'Processing TSVs...'
wordcount = Counter()
usercount = Counter()
skip_files = 0
dialog_lengths = Counter()
max_sentence_lengths = {}
for dir_ in os.listdir(args.dataroot):
    subdir = os.path.join(args.dataroot, dir_)
    for tsv in os.listdir(subdir):
        if not tsv.endswith('.tsv'):
            continue
        path = os.path.join(subdir, tsv)
        pkl = os.path.join(subdir, os.path.splitext(tsv)[0] + '.pkl')
        print path, '->', pkl
        with open(path) as f:
            user_involved = set()
            wordcount_in_file = []
            speaker_list = []
            addressee_list = []
            word_list = []
            prev_speaker = None
            prev_addressee = None
            max_sentence_length = 0
            for l in f:
                _, speaker, addressee, sentence = l.strip('\n').split('\t', 3)
                user_involved.add(speaker)
                words = nltk.word_tokenize(sentence.decode('utf-8').lower())
                wordcount_in_file.extend(words)

                if speaker == prev_speaker:
                    if prev_addressee == '':
                        addressee_list[-1] = addressee
                        prev_addressee = addressee
                    assert addressee == prev_addressee or addressee == ''
                    sentence_length += len(words)
                else:
                    sentence_length = len(words)
                    speaker_list.append(speaker)
                    addressee_list.append(addressee)
                    word_list.append([])
                word_list[-1].extend(words)
                prev_speaker = speaker
                prev_addressee = addressee
                max_sentence_length = max(max_sentence_length, sentence_length)
            if len(user_involved) < 2:
                print 'Skipping %s' % path
                skip_files += 1
                continue
            #assert len(user_involved) == 2
            usercount.update(user_involved)
            wordcount.update(wordcount_in_file)
            turns = len(speaker_list)
            dialog_lengths.update([turns])
            max_sentence_lengths[turns] = max(max_sentence_lengths.get(turns, 0), max_sentence_length)
            assert len(speaker_list) == len(addressee_list) == len(word_list)

        with open(pkl, 'wb') as f:
            pickle.dump({
                'speaker': speaker_list,
                'addressee': addressee_list,
                'words': word_list,
                }, f)

print 'Dumping word counts and user counts...'
with open('wordcount.pkl', 'wb') as f:
    pickle.dump(wordcount, f)
with open('usercount.pkl', 'wb') as f:
    pickle.dump(usercount, f)
with open('turncount.pkl', 'wb') as f:
    pickle.dump(dialog_lengths, f)
with open('max_sentence_lengths.pkl', 'wb') as f:
    pickle.dump(max_sentence_lengths, f)
print 'Skipped %d files in total' % skip_files
