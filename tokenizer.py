# Usage: python3 tokenizer.py <input.txt >output.txt
import nltk
import sys
import pickle
import os

for fname in os.listdir(sys.argv[1]):
    speaker = []
    addressee = []
    words = []
    f = open(os.path.join(sys.argv[1], fname))
    for _l in f:
        l = _l.strip().split('::')
        u, s = l
        s = ' '.join(s.split())
        s = nltk.word_tokenize(s)
        s = [w.lower() for w in s]
        speaker.append(u)
        addressee.append('')
        words.append(s)

    with open(os.path.join(sys.argv[2], os.path.splitext(fname)[0] + '.pkl'), 'wb') as f:
        pickle.dump({'speaker': speaker, 'addressee': addressee, 'words': words}, f)
