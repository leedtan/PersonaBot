import argparse
import torch
import os
import re
import h5py
import numpy as np

parser = argparse.ArgumentParser(description='Ubuntu Dialogue dataset parser')
parser.add_argument('--dataroot', type=str, required=True, help='Root of the data downloaded from github')
parser.add_argument('--outputdir', type=str, required=True, help='output directory')
parser.add_argument('--traintestsplit', type=float, default=0.8, help='train test split (default 0.8)')
parser.add_argument('--wordunknownedthreshold', type=int, default=5, help='train test split (default 0.8)')
args = parser.parse_args()
import pickle

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = ['::padding::']#Reserve first location for padding

    def add_word(self, word):
        self.counter = 1
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = self.counter # 0 is usually reservered for padding
            self.counter += 1

    def __len__(self):
        return len(self.idx2word)

class DictionaryLearner(object):
    def __init__(self):
        self.word2count = {}

    def add_word(self, word):
        self.word2count[word] = self.word2count.get(word, 0) + 1

    def getFrequentWords(self):
        self.frequentWordBank = set()
        for word in self.word2count.keys():
            if self.word2count.get(word) >= args.wordunknownedthreshold:
                self.frequentWordBank.add(word)

        print("There are : %d frequent words" % (len(self.frequentWordBank)))

        return list(self.frequentWordBank)

class Corpus(object):
    def __init__(self, path, list_dir):
        self.train_file = h5py.File(args.outputdir + 'dataset_train.h5', 'w')
        self.val_file = h5py.File(args.outputdir + 'dataset_val.h5', 'w')
        self.test_file = h5py.File(args.outputdir + 'dataset_test.h5', 'w')
        self.dicLearner = DictionaryLearner()
        self.userbank = DictionaryLearner()
        train_test_split = int(round(len(list_dir) * args.traintestsplit))
        
        train_set = list_dir[:train_test_split]
        val_set = list_dir[train_test_split:-2]
        test_set = list_dir[-2:]
        self.frequent_words, self.frequent_users = self.learnDictionnary(path, train_set)
        
        self.word2idx, self.idx2word = self.get_word_mapping(self.frequent_words)
        self.user2idx, self.idx2user = self.get_word_mapping(self.frequent_users)
        self.users = self.user2idx.keys()
        self.words = self.word2idx.keys()
        self.train = self.tokenize(path,train_set, self.train_file)
        self.valid = self.tokenize(path,val_set, self.val_file)
        self.test = self.tokenize(path,test_set, self.test_file)
        pickle.dump([self.word2idx, self.idx2word], open(args.outputdir +  "word_dicts.p", "w" ))
        pickle.dump([self.user2idx, self.idx2user], open(args.outputdir +  "user_dicts.p", "w" ))
    
    def get_word_mapping(self, freq_words):
        word2idx = {}
        idx2word = {}
        word2idx[-1] = -1
        idx2word[-1] = -1
        for idx, wd in enumerate(freq_words):
            word2idx[wd] = idx+1
            idx2word[idx+1] = wd
        return word2idx, idx2word
        
        

    def learnDictionnary(self, path, list_dir):
        for dir_name in list_dir:
            for fname in os.listdir(path + '/' + dir_name):
                with open(path + '/' + dir_name + '/' + fname) as f:
                    for line in f:
                        # @todo add date here if we want ?
                        date_user_test = re.split(r'\t+', line)
                        self.userbank.add_word(date_user_test[1])

                        words = date_user_test[2].split() + ['<eos']
                        for word in words:
                            self.dicLearner.add_word(word)

        freq_words = self.dicLearner.getFrequentWords()
        freq_users = self.userbank.getFrequentWords()
        return freq_words, freq_users

    def parseToIdx(self, pathIn, pathOut):
        with open(pathIn, 'r') as f:
            for line in f:
                pass

    def tokenize(self, root_path, list_dir, file):
        """Tokenizes a text file."""
        # Add words to the dictionary
        file_idx = 0
        for dir in list_dir:
            path = root_path  + dir + '/'
            assert os.path.exists(path)
            # Tokenize file content
            for fname in os.listdir(path):
                with open(path + fname, 'r') as f:
                    conversation = []
                    for line in f:
                        try:
                            data, user, text = re.split(r'\t+', line)
                        except:
                            continue
                        processed_text = []
                        for word in text:
                            if word in self.words:
                                processed_text.append(self.word2idx[word])
                            else:
                                processed_text.append(-1)
                        if user in self.users:
                            usr = self.user2idx[user]
                        else:
                            usr = -1
                        conversation.append(np.concatenate([np.expand_dims(np.array(usr),1), np.array(processed_text)]))
                    if len(conversation) < 5:
                        continue
                    maxlen = max([len(l) for l in conversation])
                    Z = np.zeros((len(conversation), maxlen+1))
                    for idx, row in enumerate(conversation):
                        Z[idx, :len(row)] = row
                        Z[idx,-1] = len(row)
                    file[str(file_idx)] = Z
                    file_idx += 1
        return
a = Corpus(args.dataroot, os.listdir(args.dataroot))

















