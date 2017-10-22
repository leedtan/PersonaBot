# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

with open('turncount.pkl', 'rb') as f:
    turncounts = np.array(list(pickle.load(f).values())).astype(float)

with open('max_sentence_lengths.pkl', 'rb') as f:
    sentence_lengths = np.array(list(pickle.load(f).values())).astype(float)
    
plt.close()
plt.figure(figsize=(10,10))
ax = plt.subplot(111)
#ax.set_xscale("log")
#plt.hist(turncounts,bins=np.logspace(0.1, np.log(turncounts.max()), 500))
plt.hist(turncounts,bins=100)
plt.title('turn counts')
plt.savefig('turncounts.png')
plt.close()
plt.figure(figsize=(10,10))
ax = plt.subplot(111)
#ax.set_xscale("log")
#plt.hist(sentence_lengths,bins=np.logspace(0.1, np.log(sentence_lengths.max()), 500))
plt.hist(sentence_lengths,bins=500)
plt.title('sentence lengths')
plt.savefig('sentencelengths.png')
plt.close()