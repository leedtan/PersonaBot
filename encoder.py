
import torch as T
import torch.nn as NN
from modules import *

class Encoder(NN.Module):
    '''
    Inputs:
    @wd_emb: (batch_size, max_turns, max_words, word_emb_size)
    @usr_emb: (batch_size, max_turns, user_emb_size)
    @turns: (batch_size,) LongTensor
    #sentence_lengths_padded: (batch_size, max_turns) LongTensor

    Returns:
    @encoding: Sentence encoding,
        2D (batch_size, max_turns, output_size)
    @wds_h: Annotation vectors for each word
        3D (batch_size, max_turns, max_words, output_size)
    '''
    def __init__(self,size_usr, size_wd, output_size, num_layers):
        NN.Module.__init__(self)
        self._output_size = output_size
        self._size_wd = size_wd
        self._size_usr = size_usr
        self._num_layers = num_layers

        self.rnn = NN.LSTM(
                size_usr + size_wd,
                output_size // 2,
                num_layers,
                bidirectional=True,
                )
        init_lstm(self.rnn)

    def forward(self, wd_emb, usr_emb, turns, sentence_lengths_padded):
        batch_size, max_turns, max_words, _ = wd_emb.size()
        num_layers = self._num_layers
        output_size = self._output_size

        #DBG HERE
        usr_emb = usr_emb.unsqueeze(2)
        usr_emb = usr_emb.expand(batch_size, max_turns, max_words, self._size_usr)
        #wd_emb = wd_emb.permute(1, 0, 2)
        #Concatenate these
        input_seq = T.cat((usr_emb, wd_emb), 3)
        initial_state = (
                tovar(T.zeros(num_layers * 2, batch_size * max_turns, output_size // 2)),
                tovar(T.zeros(num_layers * 2, batch_size * max_turns, output_size // 2)),
                )
        #input_seq: 93,160,26 length: 160,output_size: 18, init[0]: 2,160,9
        input_seq = input_seq.view(batch_size * max_turns, max_words, self._size_usr + self._size_wd)
        input_seq = input_seq.permute(1,0,2)
        sentence_lengths_padded = sentence_lengths_padded.view(-1)
        output, (h, c) = dynamic_rnn(self.rnn, input_seq, sentence_lengths_padded, initial_state)
        h = h.permute(1, 0, 2)
        output = output.permute(1, 0, 2)
        h = h[:, -2:].contiguous().view(batch_size * max_turns, output_size)

        h = h.view(batch_size, max_turns, -1)
        output = output.view(batch_size, max_turns, max_words, -1)
        sentence_lengths_padded = sentence_lengths_padded.view(batch_size, max_turns)
        # TODO mask out @h and @output
        h = h * mask_3d(h.size(), turns, False)
        output = output * mask_4d(output.size(), turns, sentence_lengths_padded, False)

        return h, output
