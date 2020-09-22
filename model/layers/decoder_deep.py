import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math

USE_CUDA = True


class CosineLinear(nn.Module):

    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  # for initializaiton of sigma

    def forward(self, input):

        out = F.linear(F.normalize(input, p=2, dim=1), \
                       F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out


class DecoderDeep(nn.Module):
    def __init__(self, dec_type, input_size, output_size, hidden_size, d_size, n_layer=1, dropout=0.5):
        """
        Initialize sclstm decoder
        """

        super(DecoderDeep, self).__init__()
        self.dec_type = dec_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.d_size = d_size
        self.n_layer = n_layer
        self.dropout = dropout
        self.output_dropout_layer = nn.ModuleList()
        self.last_dropout_layer = nn.ModuleList()
        self.output_dropout_distillation_layer = nn.ModuleList()
        self.last_dropout_distillation_layer = nn.ModuleList()

        print('Using sclstm as decoder with module list!', file=sys.stderr)
        assert d_size != None

        self.w2h, self.h2h = nn.ModuleList(), nn.ModuleList()
        self.w2h_r, self.h2h_r = nn.ModuleList(), nn.ModuleList()
        self.dc = nn.ModuleList()
        print(f"Number of layer is {n_layer}")

        for i in range(n_layer):
            if i == 0:
                self.w2h.append(nn.Linear(input_size, hidden_size * 4).cuda())
                self.w2h_r.append(nn.Linear(input_size, d_size).cuda())
            else:
                self.w2h.append(nn.Linear(input_size + i * hidden_size, hidden_size * 4).cuda())
                self.w2h_r.append(nn.Linear(input_size + i * hidden_size, d_size).cuda())

            self.h2h.append(nn.Linear(hidden_size, hidden_size * 4).cuda())
            self.h2h_r.append(nn.Linear(hidden_size, d_size).cuda())
            self.dc.append(nn.Linear(d_size, hidden_size, bias=False).cuda())
            self.output_dropout_layer.append(nn.Dropout(p=self.dropout))
            self.last_dropout_layer.append(nn.Dropout(p=self.dropout))
            self.output_dropout_distillation_layer.append(nn.Dropout(p=self.dropout))
            self.last_dropout_distillation_layer.append(nn.Dropout(p=self.dropout))

        if 'cosine' in self.dec_type:  # using cosine normalization
            self.out = CosineLinear(hidden_size * n_layer, output_size)
        else:
            self.out = nn.Linear(hidden_size * n_layer, output_size)

    def _step(self, input_t, last_hidden, last_cell, last_dt, layer_idx):
        '''
        Feedforward for one step in one layer in sclstm
        @param input_t: (batch_size, hidden_size)
        @param last_hidden: (batch_size, hidden_size)
        @paramlast_cell: (batch_size, hidden_size)
        @return cell, hidden, dt at this time step, all: (batch_size, hidden_size)
        '''

        # Get all gates
        w2h = self.w2h[layer_idx](input_t)  # (batch_size, hidden_size*4)
        w2h = torch.split(w2h, self.hidden_size, dim=1)  # (batch_size, hidden_size) * 4
        h2h = self.h2h[layer_idx](last_hidden[layer_idx])
        h2h = torch.split(h2h, self.hidden_size, dim=1)

        gate_i = F.sigmoid(w2h[0] + h2h[0])  # (batch_size, hidden_size)
        gate_f = F.sigmoid(w2h[1] + h2h[1])
        gate_o = F.sigmoid(w2h[2] + h2h[2])

        # Updata dt
        alpha = 1. / self.n_layer

        _gate_r = 0
        for i in range(self.n_layer):
            _gate_r += alpha * self.h2h_r[i](last_hidden[i])
        gate_r = F.sigmoid(self.w2h_r[layer_idx](input_t) + _gate_r)

        dt = gate_r * last_dt

        cell_hat = F.tanh(w2h[3] + h2h[3])
        cell = gate_f * last_cell + gate_i * cell_hat + F.tanh(self.dc[layer_idx](dt))
        hidden = gate_o * F.tanh(cell)

        return hidden, cell, dt

    def rnn_step(self, vocab_t, last_hidden, last_cell, last_dt, gen=False):
        '''
        run a step over all layers in sclstm
        '''

        cur_hidden, cur_cell, cur_dt = [], [], []
        output_hidden = []

        for i in range(self.n_layer):
            # prepare input_t
            if i == 0:
                input_t = vocab_t
                assert input_t.size(1) == self.input_size
            else:
                pre_hidden = torch.cat(output_hidden, dim=1)
                input_t = torch.cat((vocab_t, pre_hidden), dim=1)
                assert input_t.size(1) == self.input_size + i * self.hidden_size

            _hidden, _cell, _dt = self._step(input_t, last_hidden, last_cell[i], last_dt[i], i)
            cur_hidden.append(_hidden)
            cur_cell.append(_cell)
            cur_dt.append(_dt)
            if gen:
                output_hidden.append(_hidden.clone())
            else:
                output_hidden.append(self.output_dropout_layer[i](_hidden.clone()))
        last_hidden, last_cell, last_dt = cur_hidden, cur_cell, cur_dt

        if not gen:
            for i in range(self.n_layer):
                last_hidden[i] = self.last_dropout_layer[i](last_hidden[i])
        output = self.out(torch.cat(last_hidden, dim=1))
        return output, last_hidden, last_cell, last_dt

    def forward(self, input_var, dataset, init_hidden=None, init_feat=None, gen=False, sample_size=1, keep_last=False):
        '''
        Forward pass decoder \n
        @param input_var(torch.Tensor): (batch_size, max_len, emb_size) \n
        @param datast(DatasetWoz3) \n
        @param init_hidden(torch.Tensor): (batch_size, hidden_size) if exist \n
        @param init_feat(torch.Tensor): (batch_size, feat_size) if exist \n
        @param gen(boolean): generation mode or not \n
        @param sample_size(int) \n
        @param keep_last(boolean): keep last hidden for computing exemplars by herding or not \n
        @return output_prob: (batch_size, max_len, output_size) \n
        @return decoder_words \n
        @return last_hidden
        '''

        batch_size = input_var.size(0)
        max_len = 60 if gen else input_var.size(1)

        self.output_prob = Variable(torch.zeros(batch_size, max_len, self.output_size))
        if USE_CUDA:
            self.output_prob = self.output_prob.cuda()

        # container for last cell, hidden for each layer
        # NOTE: for container, using just list instead of creating a torch variable causing inplace operation runtime error
        last_hidden, last_cell, last_dt = [], [], []
        for i in range(self.n_layer):
            last_hidden.append(init_hidden.clone())
            last_cell.append(init_hidden.clone())
            last_dt.append(init_feat.clone())

        # Run rnn to decode words
        decoded_words = ['' for k in range(batch_size)]
        vocab_t = self.get_onehot('SOS_token', dataset, batch_size=batch_size)
        for t in range(max_len):
            output, last_hidden, last_cell, last_dt = self.rnn_step(vocab_t, last_hidden, last_cell, last_dt, gen=gen)

            self.output_prob[:, t, :] = output
            previous_out = self.logits2words(output, decoded_words, dataset, sample_size)
            vocab_t = previous_out if gen else input_var[:, t, :]  # (batch_size, output_size)

        if gen:
            decoded_words = self.truncate(decoded_words)

        last_hidden = torch.stack(last_hidden).squeeze() if keep_last else None
        return self.output_prob, decoded_words, last_hidden

    def truncate(self, decoded_words):
        """
        Truncate decoded words at End Of Sentence token
        """
        res = []
        for s in decoded_words:
            s = s.split()
            idx = s.index('EOS_token') if 'EOS_token' in s else len(s)
            res.append(' '.join(s[:idx]))
        return res

    def get_onehot(self, word, dataset, batch_size=1):
        """
        Get one hot representation of word
        """
        res = [[1 if index == dataset.word2index[word] else 0 for index in range(self.input_size)] for b in
               range(batch_size)]
        res = Variable(torch.FloatTensor(res))
        if USE_CUDA:
            res = res.cuda()
        return res  # (batch_size, input_size)

    def logits2words(self, output, decoded_words, dataset, sample_size):
        """
        * Decode words from logits output at a time step AND put decoded words in final results *
        * take argmax if sample size == 1
        """
        batch_size = output.size(0)
        if sample_size == 1:  # take argmax directly w/o sampling
            topv, topi = F.softmax(output, dim=1).data.topk(1)  # both (batch_size, 1)

        else:  # sample over word distribution
            topv, topi = [], []
            word_dis = F.softmax(output, dim=1)  # (batch_size, output_size)

            # sample from part of the output distribution for word variations
            n_candidate = 3
            word_dis_sort, idx_of_idx = torch.sort(word_dis, dim=1, descending=True)
            word_dis_sort = word_dis_sort[:, :n_candidate]
            idx_of_idx = idx_of_idx[:, :n_candidate]
            sample_idx = torch.multinomial(word_dis_sort, 1)  # (batch_size,)
            for b in range(batch_size):
                i = int(sample_idx[b])
                idx = int(idx_of_idx[b][i])
                prob = float(word_dis[b][idx])
                topi.append(idx)
                topv.append(prob)

            topv = torch.FloatTensor(topv).view(batch_size, 1)
            topi = torch.LongTensor(topi).view(batch_size, 1)

        decoded_words_t = np.zeros((batch_size, self.output_size))
        for b in range(batch_size):
            idx = topi[b][0].item()
            word = dataset.index2word[idx]
            decoded_words[b] += (word + ' ')
            decoded_words_t[b][idx] = 1
        decoded_words_t = Variable(torch.from_numpy(decoded_words_t.astype(np.float32)))

        if USE_CUDA:
            decoded_words_t = decoded_words_t.cuda()

        return decoded_words_t

    def beam_search(self, input_var, dataset, init_hidden=None, init_feat=None, gen=True, beam_size=1):
        """
        @param input_var: (batch_size, max_len, emb_size) \n
        @param dataset(DatasetWoz3) \n
        @param init_hidden: (batch_size, hidden_size) if exist \n
        @param init_feat: (batch_size, feat_size) if exist \n
        @param gen(boolean): generation mode or not
        @param beam_size(int) \n
        @return decoded_words: (batch_size, beam_size)
        """
        assert gen
        batch_size = dataset.batch_size
        max_len = 60  # if gen else input_var.size(1)

        # beam search data container
        init_x = {'history': ['SOS_token'], \
                  'logProb': [0], \
                  'lastStates': {'hid': [init_hidden.clone() for _ in range(self.n_layer)], \
                                 'cell': [init_hidden.clone() for _ in range(self.n_layer)], \
                                 'feat': [init_feat.clone() for _ in range(self.n_layer)] \
                                 }
                  }

        dec_words = [[init_x for _ in range(beam_size)] for _ in range(batch_size)]

        alpha = 0.7  # length normalization coefficient
        for batch_idx in range(batch_size):
            # iter over seqeuence
            for t in range(max_len):
                cand_pool = []  # pool keeps all candidates, max size: beam_size * beam_size
                # iter over history
                for beam_idx in range(beam_size):
                    beam = dec_words[batch_idx][beam_idx]
                    assert len(beam['history']) == len(beam['logProb'])

                    last_word = beam['history'][-1]
                    last_hid = beam['lastStates']['hid']
                    last_cell = beam['lastStates']['cell']
                    last_dt = beam['lastStates']['feat']

                    if last_word == 'EOS_token':
                        cand_pool.append(beam)
                    else:
                        last_word_dis = self.get_onehot(last_word, dataset, batch_size=1)
                        dis, cur_hid, cur_cell, cur_dt = self.rnn_step(last_word_dis, last_hid, last_cell, last_dt,
                                                                       gen=True)

                        dis = dis.squeeze(0)  # (bs=1, output_size) => (output_size)
                        dis = torch.log(
                            F.softmax(dis, dim=0))  # + sum(beam['logProb'])) / math.pow(1+len(beam['logProb']), alpha)
                        logProb, vocab_idx = dis.data.topk(beam_size)  # size: (beam_size)
                        # iter over candidate beams for each history
                        for cand_idx in range(beam_size):
                            print(f"The candidate idx is {vocab_idx[cand_idx]}", file=sys.stderr)

                            cand_word = dataset.index2word[vocab_idx[cand_idx].item()]

                            cand_beam = {'history': [], 'logProb': [], 'lastStates': {}}
                            cand_beam['history'] += beam['history']
                            cand_beam['history'].append(cand_word)
                            cand_beam['logProb'] += beam['logProb']
                            cand_beam['logProb'].append(logProb[cand_idx])

                            # same last states within extends from a same beam
                            cand_beam['lastStates']['hid'] = cur_hid
                            cand_beam['lastStates']['cell'] = cur_cell
                            cand_beam['lastStates']['feat'] = cur_dt
                            cand_pool.append(cand_beam)

                    if t == 0:  # can only use 1 beam to extend cuz history of all beams is the same when t == 0
                        break

                # length normalization (-1 in len if for no need to consider init logProb 0)
                cand_pool = sorted(cand_pool, key=lambda x: sum(x['logProb']) / pow(len(x['logProb']) - 1, alpha),
                                   reverse=True)
                dec_words[batch_idx] = cand_pool[:beam_size]

                if [x['history'][-1] for x in dec_words[batch_idx]] == ['EOS_token' for _ in range(beam_size)]:
                    break

        dec_words = [[' '.join(beam['history']).replace('SOS_token ', '').replace(' EOS_token', '') \
                      for beam in batch] for batch in dec_words]
        return dec_words
