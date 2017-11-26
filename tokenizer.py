# Usage: python3 tokenizer.py <input.txt >output.txt
import nltk
import sys

for _l in sys.stdin:
    l = _l.strip().split('::')
    u, s = l
    s = ' '.join(s.split())
    s = nltk.word_tokenize(s)
    s = [w.lower() for w in s]
    print('\t%s\t\t%s' % (u, ' '.join(s)))
