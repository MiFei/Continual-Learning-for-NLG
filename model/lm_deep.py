import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import copy

from model.layers.decoder_deep import DecoderDeep
from model.masked_cross_entropy import *
from model.masked_kl_divergence import masked_kl_divergence

USE_CUDA = True


class LM_deep(nn.Module):
    def __init__(self, dec_type, args, input_size, output_size, hidden_size, d_size, n_layer=1, dropout=0.5, lr=0.001):
        """
        Constructor of LM deep \n
        @param dec_type(str): decoder_type, default to sclstm \n
        @param args(argparse.ArgumentParser) \n
        @param input_size(int) \n
        @param output_size(int) \n
        @param hidden_size(int) \n
        @param d_size(int): feature vector length \n
        @param n_layer(int) \n
        @param dropout(float) \n
        @param lr(float): learning rate \n
        """

        super(LM_deep, self).__init__()
        self.dec_type = dec_type
        self.hidden_size = hidden_size

        print('Using deep version with {} layer'.format(n_layer))
        print('Using deep version with {} layer'.format(n_layer), file=sys.stderr)
        self.dec = DecoderDeep(dec_type, input_size, output_size, hidden_size, d_size=d_size, n_layer=n_layer,
                               dropout=dropout)

        self.args = args
        self.set_solver(lr)
        self.model_type = "lm"

    def forward(self, input_var, dataset, feats_var, gen=False, beam_search=False, beam_size=1, keep_last=False):
        """
        Forward pass to evaluate decoded words and last hidden (for knowledge distillation) \n
        @param input_var(torch.Tensor): input variable \n
        @param dataset(DatasetWoz3) \n
        @param feats_var(torch.Tensor): feature variable \n
        @param gen(boolean): generation mode or not \n
        @param beam_search(boolean): whether use beam search or not \n
        @param beam_size(int) \n
        @param keep_last(boolean) whether keep last hidden for computing exemplars by herding or not \n
        @return decoded_words(list)\n
        @return last_hidden(torch.Tensor): last hidden layer for compute exemplars by herding
        """

        batch_size = dataset.batch_size
        init_hidden = Variable(torch.zeros(batch_size, self.hidden_size))

        if USE_CUDA:
            init_hidden = init_hidden.cuda()

        # Decode words by going through decoder
        if beam_search:
            assert gen
            decoded_words = self.dec.beam_search(input_var, dataset, init_hidden=init_hidden, init_feat=feats_var, \
                                                 gen=gen, beam_size=beam_size)
            return decoded_words  # list (batch_size=1) of list (beam_size) with generated sentences

        # Beam search or not
        sample_size = beam_size
        decoded_words = [[] for _ in range(batch_size)]

        for sample_idx in range(sample_size):  # over generation
            self.output_prob, gens, last_hidden = self.dec(input_var, dataset, init_hidden=init_hidden,
                                                           init_feat=feats_var, \
                                                           gen=gen, sample_size=sample_size, keep_last=keep_last)
            for batch_idx in range(batch_size):
                decoded_words[batch_idx].append(gens[batch_idx])

        return decoded_words, last_hidden  # list (batch_size) of list (sample_size) with generated sentences

    def set_solver(self, lr):

        self.solver = torch.optim.Adam(self.dec.parameters(), lr=lr)

    def get_loss(self, target_label, target_lengths, do_label=None, da_label=None, sv_label=None, return_vec=False):
        """
        Compute loss = cross_entropy_loss + kl_annealing loss + do,da,sv loss \n
        @param target_label(torch.Tensor) \n
        @param target_lengths(list): list of length of target sentence \n
        @param do_label(torch.Tensor): domain labels \n
        @param da_label(torch.Tensor): dialogue act labels \n
        @param sv_label(torch.Tensor): slot value labels \n
        @param return_vec(boolean): whether return per word loss in cross entropy loss
        """

        length = Variable(torch.LongTensor(target_lengths)).cuda()
        self.loss = masked_cross_entropy(
            self.output_prob.contiguous(),  # -> batch x seq
            target_label.contiguous(),  # -> batch x seq
            length,
            return_vec=return_vec)

        return self.loss

    def add_ewc_loss(self, ewc_loss, importance):
        """
        Add ewc regularization loss to original loss
        """
        self.loss = self.loss + importance * ewc_loss

        return self.loss

    def get_distillation_loss(self, target_label, target_lengths, old_indices, ref_output,
                              new_task_num, distill_New=True, sv_ratio=1, do_label=None, da_label=None, sv_label=None):
        """
        This func calculate  distillation for all utterences in a batch
        @param target_labels: ground truth words (batch_size, max_sent_len, vocab_size)
        @param target_lengths: list of sentence lengths (va)
        @param old_indices_mask: a mask to indicate whether voc in a sentence is in old_indices, we only ditill them
        @param old_indices: voc indices that are not in this domain, we only compute kl-div w.r.t. distribution on them
        @param word_dists: unnormalized word output proba distribution, shape = (num_distillation, max_sent_len, vocab_size)
        @param new_task_num: number of sample from new task
        @param distill_New: whether to distill new utterances in the current domain
        @return distillation loss
        """

        length = Variable(torch.LongTensor(target_lengths)).cuda()
        batch_size = len(target_lengths)

        # Compute ce on all sentences
        if distill_New:
            loss_ce = masked_cross_entropy(
                self.output_prob.contiguous(),  # -> batch x seq
                target_label.contiguous(),  # -> batch x seq
                length)
        else:
            loss_ce = masked_cross_entropy(
                self.output_prob[: new_task_num].contiguous(),  # -> batch x seq
                target_label[: new_task_num].contiguous(),  # -> batch x seq
                length[: new_task_num])

        if distill_New:
            # Distill all sentences in the batch
            distillation_loss = masked_kl_divergence(self.output_prob[:, :, old_indices].contiguous(), \
                                                     ref_output[:, :, old_indices].contiguous(), \
                                                     length, self.args.T)
        else:
            # Distill exemplars only
            if new_task_num < batch_size:
                distillation_loss = masked_kl_divergence(self.output_prob[new_task_num:].contiguous(), \
                                                         ref_output[new_task_num:].contiguous(), \
                                                         length[new_task_num:], self.args.T)
            else:
                distillation_loss = torch.tensor(0)

        if self.args.adaptive:
            self.loss = loss_ce + self.args._lambda * sv_ratio * distillation_loss
        else:
            self.loss = loss_ce + self.args._lambda * distillation_loss

        return self.loss

    def update(self, clip, freeze_indices=[]):
        """
        Update model
        @param clip(float): update regularization clipping parameter \n
        @param freeze_indices(list): list of output indices to freeze in incremental learning setting
        """

        # Back prop
        self.loss.backward()

        # Clip gradient norms
        _ = torch.nn.utils.clip_grad_norm(self.dec.parameters(), clip)

        if freeze_indices:
            self.dec.out.weight.grad[freeze_indices, :] = 0
            self.dec.out.bias.grad[freeze_indices] = 0

        # Update
        self.solver.step()

        # Zero grad
        self.solver.zero_grad()
