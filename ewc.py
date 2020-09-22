"""
Implementation of Elastic weight consolidation object
"""

import torch
from torch import nn
from torch.autograd import Variable
import copy
from model.masked_cross_entropy import *

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EWC(object):
    def __init__(self, model: nn.Module, dataset: dict):

        self.model = model
        # the data we use to compute fisher information of ewc (old_exemplars)
        self.dataset = dataset
        self.dataset.reset()

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {} # previous parameters
        self._precision_matrices = self._diag_fisher() # approximated diagnal fisher information matrix

        for n, p in copy.deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):

        self.model.train()
        precision_matrices = {}
        for n, p in copy.deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.dataset.batch_size = 1  # set batch_size to 1 in ewc

        for i in range(len(self.dataset.data['train'])):
            self.model.zero_grad()
            input_var, label_var, feats_var, lengths, refs, featStrs, sv_indexes, _, do_label, da_label, sv_label = self.dataset.next_batch("train")

            # feedforward and calculate loss
            if self.model.model_type == "lm":
                decoded_words, _ = self.model(input_var, self.dataset, feats_var)
            else:
                self.model.set_prior(False)
                target_var = input_var.clone()
                decoded_words, _ = self.model(input_var, input_lengths = lengths, target_seq = target_var, target_lengths = lengths, conds_seq = feats_var, dataset = self.dataset)

            length = Variable(torch.LongTensor(lengths)).cuda()

            # empirical Fisher if we provide ground truth label
            loss = masked_cross_entropy(
                self.model.output_prob.contiguous(),  # -> batch x seq
                label_var.contiguous(),  # -> batch x seq
                length)

            loss.backward()

            for n, p in self.model.named_parameters():

                # Jump over layers that is not trained
                if p.grad is None:
                    continue
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset.data['train'])

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()

        return loss
