import sys
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

from model.layers.encoder import Encoder
from model.layers.decoder_deep import DecoderDeep
from model.model_util import sample_gaussian
from model.masked_kl_divergence import masked_kl_divergence
from model.masked_cross_entropy import *
USE_CUDA = True

class CVAE(nn.Module):

	def __init__(self, dec_type, args, hidden_size, vocab_size, latent_size, d_size, do_size, da_size, sv_size, std, n_layer=1, dropout=0.5, use_prior=False, lr=0.001, overgen=1):
		"""
		Constructor of CVAE \n
		@param dec_type(str): decoder_type, default to sclstm \n
		@param args(argparse.ArgumentParser) \n
		@param hidden_size(int) \n
		@param vocab_size(int) \n
		@param latent_size(int) \n
		@param d_size(int): feature vector length \n
		@param do_size(int): number of domain \n
		@param da_size(int): number of dialogue act \n
		@param sv_size(int): number of slot values \n
		@param std(float): std of gaussian to sample z \n
		@param n_layer(int) \n
		@param dropout(float) \n
		@param use_prior(boolean): whether use prior to generate z \n
		@param lr(float): learning rate \n
		@param overgen(int): beam search or not
		"""

		super(CVAE, self).__init__()
		self.dec_type = dec_type
		self.args = args
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.latent_size = latent_size
		self.d_size = d_size
		self.do_size = do_size
		self.dropout = dropout
		self.n_layers = n_layer
		self.lr = lr
		self.use_prior = use_prior
		self.std = std
		self.model_type = "cvae"

		# Set encoder and decoder
		self.enc = Encoder( vocab_size, hidden_size, n_layer, dropout = dropout )
		self.dec = DecoderDeep(dec_type, vocab_size, vocab_size, hidden_size, d_size, n_layer = n_layer, dropout = dropout)

		# Set recognition network
		self.recog = nn.Linear(hidden_size*n_layer*2+d_size, latent_size*2) # first 2 for bi-directional encoder, second 2 for mean and logvar

		# Set prior network
		self.fc = nn.Linear(d_size, latent_size*2)
		self.prior = nn.Linear(latent_size*2, latent_size*2)

		# Set domain, dialogue act, slot value network
		self.pred_do = nn.Linear(latent_size, do_size)
		self.pred_da = nn.Linear(latent_size, da_size)
		self.pred_sv = nn.Linear(latent_size, sv_size)

		# Initialize global time stamp for KL annealing
		self.global_t = 0

		# Set whether use random sample or not
		self.random_sample = False if overgen == 1 else True

		# Initialize solver
		self.set_solver(lr)
		self.criterion = {'xent': torch.nn.CrossEntropyLoss(), 'multilabel': torch.nn.MultiLabelSoftMarginLoss()}

	def set_prior(self, use_prior):
		"""
		Set whether use prior network to generate z
		"""
		self.use_prior = use_prior

	def set_solver(self, lr):
		self.params = [ {'params': self.enc.parameters()}, {'params': self.dec.parameters()}, \
				{'params': self.recog.parameters()}, \
				{'params': self.fc.parameters()}, {'params': self.prior.parameters()}, \
				{'params': self.pred_do.parameters()}, \
				{'params': self.pred_da.parameters()}, {'params': self.pred_sv.parameters()} ]

		self.solver = torch.optim.Adam(self.params, lr=lr)


	def gaussian_kld(self):
		"""
		Compute KL divergence between recog network and prior network
		"""

		kld = -0.5 * torch.sum(1 + (self.recog_logvar - self.prior_logvar) 
									- torch.pow(self.prior_mu - self.recog_mu, 2) / torch.exp(self.prior_logvar)
									- torch.exp(self.recog_logvar) / torch.exp(self.prior_logvar), dim=1)
		return kld


	def get_loss(self, target_label, target_lengths, do_label, da_label, sv_label, full_kl_step = 5000, return_vec = False):
		"""
		Compute loss = cross_entropy_loss + kl_annealing loss + do,da,sv loss \n
		@param target_label(torch.Tensor) \n
		@param target_lengths(list): list of length of target sentence \n
		@param do_label(torch.Tensor): domain labels \n
		@param da_label(torch.Tensor): dialogue act labels \n
		@param sv_label(torch.Tensor): slot value labels \n
		@param full_kl_step(int): kl annealing parameter \n
		@param return_vec(boolean): whether return per word loss in cross entropy loss 
		"""

		length =  Variable(torch.LongTensor(target_lengths)).cuda()

		# Compute cross entropy loss
		rc_loss = masked_cross_entropy(
			self.output_prob.contiguous(), # -> batch x seq
			target_label.contiguous(), # -> batch x seq
			length,
			return_vec = return_vec)

		# Compute kl annealing loss
		kl_weight = min(self.global_t/full_kl_step, 1.0)
		kl_loss = torch.mean(self.gaussian_kld())

		# Compute domain, dialogue act, slot value loss
		do_loss = self.criterion['xent'](self.do_output, do_label)
		da_loss = self.criterion['multilabel'](self.da_output, da_label)
		sv_loss = self.criterion['multilabel'](self.sv_output, sv_label)

		self.loss = rc_loss + kl_weight * kl_loss + do_loss + da_loss + sv_loss

		return self.loss

	def add_ewc_loss(self, ewc_loss, importance):
		"""
		Add ewc regularization loss to original loss
		"""

		self.loss = self.loss + importance * ewc_loss

		return self.loss

	def get_distillation_loss(self, target_label, target_lengths, old_indices_mask, old_indices, ref_output, new_task_num, distill_New = True, sv_ratio=1, do_label = None, da_label = None, sv_label = None, full_kl_step = 5000):
		"""
		This func calculate  distillation for all utterences in a batch \n
		@param target_labels: ground truth words (batch_size, max_sent_len, vocab_size) \n
		@param target_lengths: list of sentence lengths (va) \n
		@param old_indices_mask: a mask to indicate whether voc in a sentence is in old_indices, we only ditill them \n
		@param old_indices: voc indices that are not in this domain, we only compute kl-div w.r.t. distribution on them \n
		@param word_dists: unnormalized word output proba distribution, shape = (num_distillation, max_sent_len, vocab_size) \n
		@param new_task_num: number of sample from new task \n
		@param distill_New: whether to distill new utterances in the current domain \n
		@return distillation loss 
		"""

		length = Variable(torch.LongTensor(target_lengths)).cuda()
		batch_size = len(target_lengths)

		# Compute cross entropy loss
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

		# Compute kl annealing loss
		kl_weight = min(self.global_t/full_kl_step, 1.0)
		kl_loss = torch.mean(self.gaussian_kld())

		# Compute domain, dialogue act and slot value loss
		do_loss = self.criterion['xent'](self.do_output, do_label)
		da_loss = self.criterion['multilabel'](self.da_output, da_label)
		sv_loss = self.criterion['multilabel'](self.sv_output, sv_label)

		if distill_New:
			# Distill all sentences
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

		# Calculate loss in adaptive way or not
		if self.args.adaptive:
			self.loss = loss_ce + self.args._lambda * sv_ratio * distillation_loss + kl_weight * kl_loss + do_loss + da_loss + sv_loss
		else:
			self.loss = loss_ce + self.args._lambda * distillation_loss + kl_weight * kl_loss + do_loss + da_loss + sv_loss

		return self.loss

	def update(self, clip, freeze_indices = []):
		"""
		Update parameters
		@param clip(float): regularization clip \n
		@param freeze_indices(list): list of indices to freeze
		"""

		# Back prop
		self.loss.backward()

		# Clip gradient norms
		for p in self.params:
			_ = torch.nn.utils.clip_grad_norm_(p['params'], clip)

		if freeze_indices:
			self.dec.out.weight.grad[freeze_indices,:] = 0
			self.dec.out.bias.grad[freeze_indices]=0

		# Update
		self.solver.step()

		# Zero grad
		self.solver.zero_grad()

	def forward(self, input_seq, input_lengths, target_seq, target_lengths, conds_seq, dataset, gen=False, keep_last = False):
		"""
		Forward pass to evaluate decoded words and last hidden (for knowledge distillation) \n
		@param input_seq(torch.Tensor): input variable \n
		@param input_lengths(list): length of input sentences \n
		@param target_seq(torch.Tensor): target variable \n
		@param target_lengths(list): length of target sentences \n
		@param conds_seq(torch.Tensor): feature vectors \n
		@param dataset(DatasetWoz3) \n
		@param gen(boolean): generation mode or not \n
		@return decoded_words(list)\n
		@return last_hidden(torch.Tensor): last hidden layer for compute exemplars by herding 
		"""

		# Convert input lengths and target lenths to tensor
		input_lengths = torch.as_tensor(input_lengths, dtype = torch.int64, device = "cpu")
		target_lengths = torch.as_tensor(target_lengths, dtype = torch.int64, device = "cpu")

		# Get batch size and maximal sentence length
		batch_size = input_seq.size(0)
		max_len_enc = input_seq.size(1)

		# Run words through encoder
		_, encoder_hidden = self.enc(input_seq, input_lengths) # (n_layers*n_directions, batch_size, hidden_size)

		l = torch.split(encoder_hidden, 1, dim=0) # a list of tensor (1, batch_size, hidden_size) with len=n_layers*n_directions
		encoder_hidden = torch.cat(l, dim=2).squeeze(dim = 0) # (batch_size, hidden_size*n_layers*n_directions)

		# Run recognition network
		recog_input = torch.cat((encoder_hidden, conds_seq), dim=1)
		recog_mulogvar = self.recog(recog_input) # (batch_size, latent_size*2)
		self.recog_mu, self.recog_logvar = torch.split(recog_mulogvar, self.latent_size, dim=1)

		# Run prior network
		prior_fc = torch.tanh(self.fc(conds_seq))
		prior_mulogvar = self.prior(prior_fc)
		self.prior_mu, self.prior_logvar = torch.split(prior_mulogvar, self.latent_size, dim=1)

		# Draw latent sample
		z = sample_gaussian(self.prior_mu, self.prior_logvar, self.std) if self.use_prior else sample_gaussian(self.recog_mu, self.recog_logvar, self.std) # (batch_size, latent_size)
		self.z = z
	
		# Predict domain, dialogue act and slot values
		self.do_output = self.pred_do(z)
		self.da_output = self.pred_da(z)
		self.sv_output = self.pred_sv(z)

		# prepare decoder s0 = wi*[c,z]+bi
		last_hidden = z

		# Decode 
		self.output_prob, gens, _ = self.dec(target_seq, dataset, init_hidden = last_hidden, init_feat = conds_seq, gen=gen, keep_last = keep_last)
		
		decoded_words = [ [] for _ in range(batch_size)]
		for batch_idx in range(batch_size):
			decoded_words[batch_idx].append(gens[batch_idx])

		return decoded_words, last_hidden
