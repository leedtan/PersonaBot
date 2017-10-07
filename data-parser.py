import argparse
import torch
import os
import re
import h5py

parser = argparse.ArgumentParser(description='Ubuntu Dialogue dataset parser')
parser.add_argument('--dataroot', type=str, required=True, help='Root of the data downloaded from github')
parser.add_argument('--outputdir', type=str, required=True, help='output directory')
parser.add_argument('--traintestsplit', type=float, default=0.8, help='train test split (default 0.8)')
parser.add_argument('--wordunknownedthreshold', type=int, default=5, help='train test split (default 0.8)')
args = parser.parse_args()

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class DictionaryLearner(object):
    def __init__(self):
        self.word2count = {}

    def add_word(self, word):
        self.word2count[word] = self.word2count.get(word, 0) + 1

    def getFrequentWords(self):
        frequentWordBank = Dictionary()
        for word in self.word2count.keys():
            if self.word2count.get(word) >= args.wordunknownedthreshold:
                frequentWordBank.add_word(word)

        print("There are : %d frequent words" % (len(frequentWordBank)))

        return frequentWordBank

class Corpus(object):
    def __init__(self, path, list_dir):
        self.userbank = Dictionary()
        train_test_split = round(len(list_dir) * args.traintestsplit)
        self.wordbank = self.learnDictionnary(path, list_dir[:train_test_split])

        self.train = self.tokenize(path, list_dir[:train_test_split])
        self.valid = self.tokenize(path, list_dir[train_test_split:])

    def learnDictionnary(self, path, list_dir):
        dicLearner = DictionaryLearner()
        for dir_name in list_dir:
            for fname in os.listdir(path + '/' + dir_name):
                with open(path + '/' + dir_name + '/' + fname) as f:
                    for line in f:
                        # @todo add date here if we want ?
                        date_user_test = re.split(r'\t+', line)
                        self.userbank.add_word(date_user_test[1])

                        words = date_user_test[2].split() + ['<eos']
                        for word in words:
                            dicLearner.add_word(word)

        return dicLearner.getFrequentWords()

    def parseToIdx(self, pathIn, pathOut):
        with open(pathIn, 'r') as f:
            for line in f:


    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return

