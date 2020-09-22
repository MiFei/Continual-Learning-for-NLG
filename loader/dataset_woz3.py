import os
import re
import sys
import json
import random
import numpy as np
import torch
import configparser
from torch.autograd import Variable
from collections import defaultdict

USE_CUDA = True

class DatasetWoz3(object):
	"""
	data container for woz dataset
	"""

	def __init__(self, config, dataSplit_file=None, percentage=1.0, data = None, exemplars = None):
		"""
		Construct dataset object \n
		@param: \n
			config(configparser.ConfigParser): configuration object \n
			dataSplit_file(str): location to load data \n
			percentage(float): Percentage of training samples to use \n
			data(list): Manually assigned data, default to None \n
			exemplars(dict): Exemplars 
		"""
		# Get configuration
		feat_file = config["DATA"]["feat_file"]
		text_file = config["DATA"]["text_file"]
		vocab_file = config["DATA"]["vocab_file"]
		template_file = config["DATA"]["template_file"]

		# Set configuration object
		self.config = config

		# Set template file to obtain domain, dialogue act and slot values
		self.template = template_file
		
		# Set hyper paramters
		self.batch_size = config.getint("DATA", "batch_size")
		self.percentage = percentage 								# Training percentage, default to 1
		self.data   = {"train":[],"valid":[],"test":[]} 
		self.data_index  = {"train": 0, "valid": 0, "test": 0} 		# Index for accessing data, used for reordering
		self.n_batch = {}
		self.shuffle = config.getboolean("DATA", "shuffle")

		# Load vocabulary
		self._loadVocab(vocab_file)
		
		# Set input feature cardinality
		self._setCardinality(template_file)
		self.do_size = self.dfs[1] - self.dfs[0]
		self.da_size = self.dfs[2] - self.dfs[1]
		self.sv_size = self.dfs[3] - self.dfs[2]
		
		# Initialize dataset
		# Here we allow directly initialize from data for generating task in incremental setting
		if dataSplit_file != None:
			self._setupData(text_file, feat_file, dataSplit_file)
		else:
			self.data = data
			self.n_batch = dict(zip(self.data.keys(), [len(self.data[key]) // self.batch_size for key in self.data.keys()]))

		# Set granularity of dataset
		# 1 represent dialogue act based and 0 represent domain based
		self.granularity = config.getint("DATA", "granularity")

		# Set exemplar file
		self.exemplars = exemplars
		
		# Reset index of dataset to zero
		self.reset()

 
	def reset(self, custom_shuffle = True):
		"""
		Reset index of data to zero, shuffle if necessary
		"""
		self.data_index  = {"train": 0, "valid": 0, "test": 0}
		if self.shuffle and custom_shuffle:
			random.shuffle(self.data["train"])

	def reset_batch_size(self, batch_size):
		"""
		Reset batch size
		"""
		for key in self.n_batch.keys():

			self.n_batch[key] = len(self.data[key]) // batch_size
		
		self.batch_size = batch_size

		# Reset dataindices
		self.reset(custom_shuffle = False)

	def next_batch(self, data_type="train", selected_indices = None):
		"""
		Obtain next batch or select a batch according to selected indices \n
		@param: \n
			data_type(str): type of data to provide \n
			selected_indices(list): specified data indices to provide \n
		@return: \n
			input_var(torch.Tensor): input variable to model \n
			label_var(torch.Tensor): target variable \n
			feats_var(torch.Tensor): features variable (d_vec) \n
			lengths(list): lengths of sentences \n
			refs(list): delexicalized sentences \n
			featStrs(list): feature strings \n
			sv_indexes(list): list of slot value indices \n
			meta(list): meta info of sentences \n
			do_label(torch.Tensor): domain tensor \n
			da_label(torch.Tensor): dialogue act tensor \n
			sv_label(torch.Tensor): slot value tensor \n
		"""
		def indexes_from_sentence(sentence, add_eos=False):
			indexes = [self.word2index[word] if word in self.word2index else self.word2index["UNK_token"] for word in sentence.split(" ")]
			if add_eos:
				return indexes + [self.word2index["EOS_token"]]
			else:
				return indexes

		# Pad a with the PAD symbol
		def pad_seq(seq, max_length):
			seq += [self.word2index["PAD_token"] for i in range(max_length - len(seq))]
			return seq

		# turn list of word indexes into 1-hot matrix
		def getOneHot(indexes):
			res = []
			for index in indexes:
				hot = [0]*len(self.word2index)
				hot[index] = 1
				res.append(hot)
			return res

		if selected_indices is None:
			# reading a batch
			start = self.data_index[data_type]

			# TODO: Handle not-full batch for task with limited amount of data

			end = self.data_index[data_type] + self.batch_size
			data = self.data[data_type][start:end]
			self.data_index[data_type] += self.batch_size
			indexes = [i for i in range(start, end)]
		else:
			data = [self.data[data_type][i] for i in selected_indices]

		sentences, refs, feats, featStrs, metas = [], [], [], [], []
		do_label, da_label, sv_label = [], [], []
		sv_indexes = []

		for dial_idx, turn_idx, text, meta in data:

			text_ori, text_delex = text["ori"], text["delex"]
			sentences.append(indexes_from_sentence(text_delex, add_eos=True))
			refs.append(text_delex)
			metas.append(meta)

			# get semantic feature
			do_idx, da_idx, sv_idx, featStr = self.getFeatIdx(meta)
			do_cond = [1 if i in do_idx else 0 for i in range(self.do_size)] # domain condition
			da_cond = [1 if i in da_idx else 0 for i in range(self.da_size)] # dial act condition
			sv_cond = [1 if i in sv_idx else 0 for i in range(self.sv_size)] # slot/value condition
			do_label.append(do_idx[0])
			da_label.append(da_cond)
			sv_label.append(sv_cond)
			feats.append(do_cond + da_cond + sv_cond)
			featStrs.append(featStr)

			sv_indexes.append(sv_idx)

		# Pad with 0s to max length
		lengths = [len(s) for s in sentences]
		max_length = 60 if self.config.getboolean("DATA", "sent_max_len") else max(lengths)
		sentences_padded = [pad_seq(s, max_length) for s in sentences]

		# Turn (batch_size, max_len) into (batch_size, max_len, n_vocab)
		sentences = [getOneHot(s) for s in sentences_padded]

		# Create torch variables
		input_var = Variable(torch.FloatTensor(sentences))
		label_var = Variable(torch.LongTensor(sentences_padded))
		feats_var = Variable(torch.FloatTensor(feats))
		do_label = Variable(torch.LongTensor(do_label))
		da_label = Variable(torch.FloatTensor(da_label))
		sv_label = Variable(torch.FloatTensor(sv_label))

		if USE_CUDA:
			input_var = input_var.cuda()
			label_var = label_var.cuda()
			feats_var = feats_var.cuda()
			do_label = do_label.cuda()
			da_label = da_label.cuda()
			sv_label = sv_label.cuda()

		return input_var, label_var, feats_var, lengths, refs, featStrs, sv_indexes, metas, do_label, da_label, sv_label

	def _setCardinality(self, template_file):
		"""
		Set cardinality of features (domain, dialogue act, slot values)
		"""
		self.cardinality = []
		with open(template_file) as f:
			self.dfs = [0,0,0,0]
			for line in f.readlines():
				self.cardinality.append(line.replace("\n",""))
				if line.startswith("d:"):
					self.dfs[1]+=1
				elif line.startswith("d-a:"):
					self.dfs[2]+=1
				elif line.startswith("d-a-s-v:"):
					self.dfs[3]+=1
			for i in range(0, len(self.dfs)-1):
				self.dfs[i+1] = self.dfs[i] + self.dfs[i+1]


	def printDataInfo(self):
		print("***** DATA INFO *****")
		print("***** DATA INFO *****", file=sys.stderr)
		print("Using {}% of training data".format(self.percentage*100))
		print("BATCH SIZE:", self.batch_size)
		
		print("Train:", len(self.data["train"]), "turns")
		print("Valid:", len(self.data["valid"]), "turns")
		print("Test:", len(self.data["test"]), "turns")
		print("# of turns", file=sys.stderr)
		print("Train:", len(self.data["train"]), file=sys.stderr)
		print("Valid:", len(self.data["valid"]), file=sys.stderr)
		print("Test:", len(self.data["test"]), file=sys.stderr)
		print("# of batches: Train {} Valid {} Test {}".format(self.n_batch["train"], self.n_batch["valid"], self.n_batch["test"]))
		print("# of batches: Train {} Valid {} Test {}".format(self.n_batch["train"], self.n_batch["valid"], self.n_batch["test"]), file=sys.stderr)
		print("*************************\n")


	def _setupData(self, text_file, feat_file, dataSplit_file):
		"""
		Load data
		"""
		with open(text_file) as f:
			dial2text = json.load(f)
		with open(feat_file) as f:
			dial2meta = json.load(f)

		with open(dataSplit_file) as f:
			dataSet_split = json.load(f)

		for data_type in ["train", "valid", "test"]:
			for dial_idx, turn_idx, _ in dataSet_split[data_type]:
				# might have empty feat turn which is not in feat file
				if turn_idx not in dial2meta[dial_idx]:
					continue

				meta = dial2meta[dial_idx][turn_idx]
				text = dial2text[dial_idx][turn_idx]
				self.data[data_type].append((dial_idx, turn_idx, text, meta))

		# percentage of training data
		if self.percentage < 1:
			_len = len(self.data["train"])
			self.data["train"] = self.data["train"][:int(_len*self.percentage)]

		# setup number of batch
		for _type in ["train", "valid", "test"]:
			self.n_batch[_type] = len(self.data[_type]) // self.batch_size

		print("Whole Data Set INFO", file = sys.stderr)
		self.printDataInfo()


	
	def _loadVocab(self,vocab_file):
		# load vocab
		self.word2index = {}
		self.index2word = {}
		idx = 0
		with open(vocab_file) as fin:
			for word in fin.readlines():
				word = word.strip().split("\t")[0]
				self.word2index[word] = idx
				self.index2word[idx] = word
				idx += 1


	def getFeatIdx(self, meta):
		feat_container = []
		do_idx, da_idx, sv_idx = [], [], []
		for da, slots in meta.items():

			do = da.split("-")[0]
			_do_idx = self.cardinality.index("d:"+do) - self.dfs[0]
			if _do_idx not in do_idx:
				do_idx.append(_do_idx)
			da_idx.append( self.cardinality.index("d-a:"+da) - self.dfs[1] )
			for _slot in slots: # e.g. ("Day", "1", "Wednesday ")
				sv_idx.append( self.cardinality.index("d-a-s-v:"+da+"-"+_slot[0]+"-"+_slot[1]) - self.dfs[2] )
				feat_container.append( da+"-"+_slot[0]+"-"+_slot[1] )

		feat_container = sorted(feat_container) # sort SVs across DAs to make sure universal order
		feat = "|".join(feat_container)

		return do_idx, da_idx, sv_idx, feat

	def get_domain_list(self):
		"""
		This function get list of domain
		"""

		with open(self.template, "r") as f:
			feature_map = f.read()
			feature_map = feature_map.split("\n")
			feature_map = np.asarray(feature_map)
			feature_map = feature_map[: 7]
		return feature_map

	def get_dialogue_act_list(self):
		"""
		This function get list of dialogue act
		"""
		
		da_list = set()
		with open(self.template, "r") as f:
			feature_map = f.read()
			feature_map = feature_map.split("\n")

			for item in feature_map:
				if item.startswith("d-a:") and "general" not in item:
					da_list.add(item.split("-")[-1])

		da_list = list(da_list)

		return da_list