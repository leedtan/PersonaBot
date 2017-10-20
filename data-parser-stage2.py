import argparse
import os
import pickle

parser = argparse.ArgumentParser(description='Ubuntu Dialogue dataset parser')
parser.add_argument('--dataroot', type=str, required=True, help='Root of the data downloaded from github')
args = parser.parse_args()

turncount = {}
max_sentence_lengths = {}
for dir_ in os.listdir(args.dataroot):
    subdir = os.path.join(args.dataroot, dir_)
    files = [f for f in os.listdir(subdir) if f.endswith('.pkl')]
    for pkl in files:
        path = os.path.join(subdir, pkl)
        relpath = os.path.join(dir_, pkl)
        print relpath
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        speaker_list = obj['speaker']
        addressee_list = obj['addressee']
        word_list = obj['words']
        turncount[relpath] = len(speaker_list)
        max_sentence_lengths[relpath] = max(len(s) for s in word_list)

with open('turncount.pkl', 'wb') as f:
    pickle.dump(turncount, f)
with open('max_sentence_lengths.pkl', 'wb') as f:
    pickle.dump(max_sentence_lengths, f)
