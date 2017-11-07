
import torch as T
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import numpy as np
from collections import Counter

EOS = '<eos>'
START = '<start>'
UNKNOWN = '<unknown>'

class UbuntuDialogDataset(Dataset):
    def __init__(self,
                 root='.',
                 wordcount_pkl='wordcount.pkl',
                 usercount_pkl='usercount.pkl',
                 turncount_pkl='turncount.pkl',
                 max_sentence_lengths_pkl='max_sentence_lengths.pkl',
                 vocab_size=159996,          # excluding padding, <unknown>, <eos>, <start>
                 user_size=None,
                 min_word_occurrence=None,
                 min_user_occurrence=5,
                 max_sentence_length_allowed=50,
                 max_turns_allowed=10,
                 coalesce_types=[],
                 ):
        '''
        Each item is a three-element key-value pair:
          * addressee
          * speaker
          * words (each element is a sequence of words)

        None of the tokenized words has the form of '<xxx>', because NLTK always
        tokenizes '<' and '>' as separate tokens.  So '<xxx>' can be used as
        special tokens, e.g. '<unknown>', '<path>', etc.
        None of the users has the form of '<xxx>', so the same applies for users
        as well.
        '''
        self.max_sentence_length_allowed = max_sentence_length_allowed
        self._pkls = []
        self._max_turns_allowed = max_turns_allowed
        with open(turncount_pkl, 'rb') as f:
            self._turncount = pickle.load(f)
        with open(max_sentence_lengths_pkl, 'rb') as f:
            self._max_sentence_lengths = pickle.load(f)
        with open(wordcount_pkl, 'rb') as f:
            self._wordcount = pickle.load(f)
        with open(usercount_pkl, 'rb') as f:
            self._usercount = pickle.load(f, encoding='windows-1252')
        for curdir in os.listdir(root):
            files = os.listdir(os.path.join(root, curdir))
            pkls = [os.path.join(root, curdir, f)
                    for f in files
                    if f.endswith('.pkl')]
            self._pkls.extend(pkls)

        if coalesce_types is not None:
            self._coalesce(coalesce_types)

        if not min_word_occurrence:
            self._vocab = [''] + list(list(zip(*self._wordcount.most_common(vocab_size)))[0])
        else:
            self._vocab = [''] + [w for w, c in self._wordcount.items() if c >= min_word_occurrence]
        self._vocab.append(START)
        self._vocab.append(EOS)

        if not min_user_occurrence:
            self._users = [''] + list(list(zip(*self._usercount.most_common(user_size)))[0])
        else:
            self._users = [''] + [u for u, c in self._usercount.items() if c >= min_user_occurrence]

        self._ivocab = {w: i for i, w in enumerate(self._vocab)}
        self._iusers = {u: i for i, u in enumerate(self._users)}

    def _coalesce(self, types):
        '''
        Right now it is treating any word starting with a '/' as a
        "<path>" token.
        Certainly inaccurate.
        Feel free to update this.
        '''
        for w in list(self._wordcount.keys()):
            if 'path' in types and w[0] == '/':
                self._wordcount['<path>'] = self._wordcount.get('<path>', 0) + self._wordcount[w]
                del self._wordcount[w]

    def __len__(self):
        return len(self._pkls)

    def __getitem__(self, i):
        return self.get_indexed_item(i)

    def get_raw_item(self, i):
        '''
        Gets a dialogue with words/users as strings
        '''
        with open(self._pkls[i], 'rb') as f:
            item = pickle.load(f)
            for i in range(len(item['words'])):
                item['words'][i] = item['words'][i][:self.max_sentence_length_allowed - 2]  # START and EOS
                item['words'][i].insert(0, START)
                item['words'][i].append(EOS)
            return item

    def get_indexed_item(self, i):
        '''
        Gets a dialogue with words/users translated as indices
        '''
        item = self.get_raw_item(i)
        addressee_idx = [self.index_user(u) for u in item['addressee']]
        speaker_idx = [self.index_user(u) for u in item['speaker']]
        word_idx = [[self.index_word(w) for w in s] for s in item['words']]
        if len(speaker_idx) > self._max_turns_allowed:
            start = np.random.randint(len(speaker_idx) - self._max_turns_allowed)
        else:
            start = 0

        addressee_idx = addressee_idx[start:start+self._max_turns_allowed]
        speaker_idx = speaker_idx[start:start+self._max_turns_allowed]
        word_idx = word_idx[start:start+self._max_turns_allowed]
        return addressee_idx, speaker_idx, word_idx

    def translate_item(self, addressee_idx, speaker_idx, word_idx):
        '''
        Translates user/word indices into strings, and remove trailing paddings
        for each sentence.
        '''
        addressee = [self.get_user(i) for i in addressee_idx] if addressee_idx is not None else None
        speaker = [self.get_user(i) for i in speaker_idx] if speaker_idx is not None else None
        words = [[self.get_word(i) for i in s] for s in word_idx] if word_idx is not None else None

        if words is not None:
            for s in words:
                while s[-1] == '':
                    s.pop(-1)

        return addressee, speaker, words

    @property
    def unknown_word_index(self):
        return len(self._vocab)

    @property
    def start_token_index(self):
        return self.index_word(START)

    @property
    def end_token_index(self):
        return self.index_word(EOS)

    @property
    def unknown_user_index(self):
        return len(self._users)

    @property
    def vocab(self):
        '''
        The vocabulary
        First word is padding; the real vocabulary starts from index 1
        '''
        return self._vocab

    @property
    def users(self):
        '''
        The users
        First user is null; the real list of users starts from index 1
        '''
        return self._users

    def get_word(self, i):
        if 0 <= i < len(self._vocab):       # padding already in self._vocab
            return self._vocab[i]
        elif i == self.unknown_word_index:
            return UNKNOWN
        else:
            raise ValueError('index out of range')

    def get_user(self, i):
        if 0 <= i < len(self._users):       # null already in self._users
            return self._users[i]
        elif i == self.unknown_user_index:
            return UNKNOWN
        else:
            raise ValueError('index out of range')

    def index_word(self, word):
        return self._ivocab[word] if word in self._ivocab else self.unknown_word_index

    def index_user(self, user):
        return self._iusers[user] if user in self._iusers else self.unknown_user_index

def collate_as_tensor(samples):
    '''
    Returns:
      turns: LongTensor of (batch_size,)
      sentence_lengths_padded: LongTensor of (batch_size, max(turns))
      speaker_padded: LongTensor of (batch_size, max(turns))
      addressee_padded: LongTensor of (batch_size, max(turns))
      words_padded: LongTensor of (batch_size, max(turns), max(sentence_lengths))
      words_reverse_padded: LongTensor of (batch_size, max(turns), max(sentence_lengths))
        * For bidirectional RNN on sentence level
    '''
    batch_size = len(samples)
    addressees, speakers, sentences = zip(*samples)

    def pad_by_turns(l, max_turns):
        return np.array([s + [0] * (max_turns - len(s)) for s in l])

    turns = np.array([len(s) for s in speakers])
    sentence_lengths = [[len(s) for s in sent] for sent in sentences]
    max_turns = np.max(turns)
    sentence_lengths_padded = pad_by_turns(sentence_lengths, max_turns)
    speaker_padded = pad_by_turns(speakers, max_turns)
    addressee_padded = pad_by_turns(addressees, max_turns)

    max_sentence_length = np.max(sentence_lengths_padded)
    words_padded = np.zeros((batch_size, max_turns, max_sentence_length), dtype=np.int64)
    words_reverse_padded = np.zeros((batch_size, max_turns, max_sentence_length), dtype=np.int64)
    for i in range(batch_size):
        for j in range(turns[i]):
            words_padded[i, j, :sentence_lengths[i][j]] = sentences[i][j]
            words_reverse_padded[i, j, :sentence_lengths[i][j]] = sentences[i][j][::-1]

    turns = T.from_numpy(turns)
    sentence_lengths_padded = T.from_numpy(sentence_lengths_padded)
    speaker_padded = T.from_numpy(speaker_padded)
    addressee_padded = T.from_numpy(addressee_padded)
    words_padded = T.from_numpy(words_padded)
    words_reverse_padded = T.from_numpy(words_reverse_padded)

    return turns, sentence_lengths_padded, speaker_padded, addressee_padded, words_padded, words_reverse_padded

class UbuntuDialogDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, num_workers=0):
        DataLoader.__init__(self,
                            dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_as_tensor,
                            num_workers=num_workers,
                            drop_last=True,
                            )

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
