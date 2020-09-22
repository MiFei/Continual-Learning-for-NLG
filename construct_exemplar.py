import sys
import math
import numpy as np
import torch
import argparse
from collections import defaultdict

from loader.task import Task
from loader.task import Exemplars

def construct_exemplar_indices_herding(x, m):
	"""
	This func return indices of m exemplars using the herding idea from input of x vectors \n
	@param: \n
		x(list): feature vectors \n
		m(int): number of exemplars to keep \n
	@return: \n
		selected_indices(list) \n
	"""

	# Initialize mean and selected ids
	if isinstance(x, list):
		print("WE HAVE LIST", file = sys.stderr)
		x = np.asarray(x)
	selected_ids = []
	mean = x.mean(axis = 0)

	for i in range(m):
		# Calculate scores of current round
		current_offset = (i / (i + 1)) * x[selected_ids, :].sum(axis = 0)
		distance = ((1 / (i + 1) * x + current_offset - mean) ** 2).sum(axis = 1)

		# Eliminate the selected ones from next candidate
		if len(selected_ids) > 0:
			offset = distance.sum() + 1
			distance[torch.tensor(selected_ids)] += offset 

		# Get closest element apart from selected ones
		selected_ids.append(distance.argmin())
	return selected_ids

def construct_exemplar_indices_loss(loss, m, sv_indexes):
	"""
	This function returns indices of m exemplars using prioritized method specified in the paper \n
	@param:
		loss(tensor.shape = (num_data,)) \n
		m(int): number of exemplars \n
		sv_indexes(list): list of sv_indexes in each sample
	@return selected_ids(list)
	"""
	sorted_loss, sorted_index = torch.sort(loss)
	selected_ids = []

	# if num_unique_sv > m: we have enough sv patterns to choose exemplars (e.g. Taxi domain on validation or m large)
	# We need multiple passes in this case
	number = min(m, len(loss))
	while len(selected_ids) < number:
		seen_sv = []
		current_loss = 0
		for itr, tmp_loss in enumerate(sorted_loss):
			if sv_indexes[sorted_index[itr]] not in seen_sv and sorted_index[itr] not in selected_ids and tmp_loss > current_loss:
				selected_ids.append(sorted_index[itr])
				current_loss = tmp_loss
				seen_sv.append(sv_indexes[sorted_index[itr]])

				if len(selected_ids) >= number:
					break
			else:
				continue # loss of current is the same

	selected_ids = torch.tensor(selected_ids).cuda()

	return selected_ids.tolist()

def construct_exemplar_indices_random(num_data, m):
	"""
	This func returns indices of m exemplars using random sampling \n
	@param: \n
		num_data(int): number of data \n
		m(int): number of exemplars to select \n
	@return: \n
		selected_ids(list): selected ids \n
	"""

	selected_ids = np.random.choice(num_data, m, replace = False)
	return selected_ids.tolist()

def construct_exemplars(model, task, exemplar_size, config, sv_len_weight, return_selected_ids = False, generation = False):
	"""
	This function construct the exemplars of current task\n
	@param:
		model(nn.Module): model to evaluate\n
		task (Task): data of task\n
		exemplar_size (dict): size of exemplar for train and validation data\n
		config (configparser.ConfigParser)\n
		sv_len_weight(float): weight of slot-value-count in finding exemplars \n
		return_selected_ids(boolean): whether return selected_ids or not \n
		generation: 'ground_truth' or 'gen' to use the model generated sent \n
	@return: \n
		exemplars (exemplars)\n
	"""

	# Read configuration
	experiment_type = config["EXPERIMENT"]["experiment"]
	exemplar_selection = config["EXPERIMENT"]["exemplar_selection"].split(",")
	batch_size = config.getint("DATA", "batch_size")

	exemplar_type = ""
	for selection_scheme in exemplar_selection:
		if selection_scheme in experiment_type:
			exemplar_type = selection_scheme
	print(f"The algorithm to select exemplars is {exemplar_type}", file = sys.stderr)
	print(f"The algorithm to select exemplars is {exemplar_type}")

	# Enter evaluation mode
	model.eval()

	with torch.no_grad():

		task.reset()
		feature_map = dict({"train": [], "valid": []})
		sv_indexes_map = dict({"train": [], "valid": []})

		""" ------ Obtain input for exemplar selection ------ """
		for dtype in ["train", "valid"]:
			for i in range(task.n_batch[dtype]):

				# Obtain next batch of data
				input_var, label_var, feats_var, lengths, refs, featStrs, sv_indexes, _, do_label, da_label, sv_label = task.next_batch(dtype)
				sv_indexes = [sorted(tmp) for tmp in sv_indexes] # do not differentatiate the order of sv pairs

				# Feed forward and get the input for exemplar selection
				if exemplar_type == "herding":

					if model.model_type == "lm":
						_, last_hidden = model(input_var, task, feats_var, keep_last=False)
					else:
						target_var = input_var.clone()
						model.set_prior(False)
						_, last_hidden = model(input_var, input_lengths = lengths, target_seq = target_var, target_lengths = lengths, conds_seq = feats_var, dataset = task)

					norm = feats_var.norm(p=1, dim=1, keepdim=True)
					feats_var_normalized = feats_var / norm

					if model.model_type == "lm":
						feature_map[dtype].append(feats_var_normalized)
					else:
						feature_map[dtype].append(last_hidden)
				elif exemplar_type == "loss":
					
					# Feed forward to get loss
					if model.model_type == "lm":
						_ = model(input_var, task, feats_var, keep_last=False)
					else:
						target_var = input_var.clone()
						model.set_prior(False)
						_ = model(input_var, input_lengths = lengths, target_seq = target_var, target_lengths = lengths, conds_seq = feats_var, dataset = task)

					loss = model.get_loss(label_var, lengths, return_vec=True, do_label = do_label, da_label = da_label, sv_label = sv_label)  # raw loss of the model
					sv_count = torch.tensor(
						[len(sv) for sv in sv_indexes]).cuda()  # number of sv in each sentence of this batch

					# Computed weighted loss
					modified_loss = loss * sv_count.pow(sv_len_weight)
					feature_map[dtype].append(modified_loss)
					sv_indexes_map[dtype] += sv_indexes  # regular list, not torch tensor

			if exemplar_type in ["herding", "loss"]:
				feature_map[dtype] = torch.cat(feature_map[dtype])

		""" ------ Get exemplars data for train and validation ------ """
		# Number of samples to pass into model
		# This dict is used only for distillation
		m = dict({	"train": math.ceil(exemplar_size["train"] / batch_size) * batch_size,
					"valid": math.ceil(exemplar_size["valid"] / batch_size) * batch_size})

		exemplars = dict()
		exemplar_data = dict()
		selected_ids = dict()

		for dtype in ["train", "valid"]:

			# Get exemplar ids using specified selection scheme
			print(f"Getting {exemplar_type} exemplars", file = sys.stderr)
			print(f"Getting {exemplar_type} exemplars")

			if exemplar_type == "herding":
				dtype_selected_ids = construct_exemplar_indices_herding(feature_map[dtype], m[dtype])
			elif exemplar_type == "loss":
				dtype_selected_ids = construct_exemplar_indices_loss(feature_map[dtype], m[dtype], sv_indexes_map[dtype])
			else:	# Random selection of exemplars
				dtype_selected_ids = construct_exemplar_indices_random(len(task.data[dtype]), m[dtype])
			
			del feature_map[dtype]

			# Get exemplars data
			if not generation:
				dtype_exemplar_data = [task.data[dtype][i] for i in dtype_selected_ids]
			else:
				# Generate pseudo exemplars using our model
				dtype_exemplar_data = list()
				
				# Get exemplars data by generation
				for i in range(len(dtype_selected_ids) // batch_size):

					# Get decoded words of current batch
					batch_ids = dtype_selected_ids[i * batch_size: (i + 1) * batch_size]
					input_var, label_var, feats_var, lengths, refs, featStrs, sv_indexes, meta, _, _, _ = task.next_batch(dtype,
																										   batch_ids)
					# Feed forward to get loss
					if model.model_type == "lm":
						decoded_words, _ = model(input_var, task, feats_var, keep_last=False)
					else:
						target_var = input_var.clone()
						model.set_prior(True)
						decoded_words, _ = model(input_var, input_lengths = lengths, target_seq = target_var, target_lengths = lengths, conds_seq = feats_var, dataset = task)

					# Remove UNK_token at the end of generated sentences
					for i, sent in enumerate(decoded_words):
						if sent[0][-9:] == "UNK_token":
							decoded_words[i] = [sent[0][: -9]]

					decoded_words = [" ".join(sent[0].split(' ')[: 59]) for sent in decoded_words]

					# Add generated current batch to exemplar
					dtype_exemplar_data.extend([[f"gen{(i + 1) * j}", f"gen{(i + 1) * j}",
												 dict({"ori": decoded_words[j], "delex": decoded_words[j]}),
												 meta[j]] for j in range(len(decoded_words))])

				print(f"Domain {task.task_name} has unique d_vec: {dtype}: {len(dtype_selected_ids)}", file=sys.stderr)
				print(f"Domain {task.task_name} has unique d_vec: {dtype}: {len(dtype_selected_ids)}")
				print(f"Generating exemplars from {exemplar_type}", file=sys.stderr)

			# Store current dtype's exemplar data and ids
			exemplar_data[dtype] = dtype_exemplar_data
			selected_ids[dtype] = dtype_selected_ids
			
		""" Get distillation if necessary """
		if 'distillation' in experiment_type:
			distillations = compute_distillation(model, task, selected_ids)
		else:
			distillations = dict({"train": None, "valid": None})

		""" Construct exemplars """
		for dtype in ["train", "valid"]:

			#if exemplar_type == 'ground_truth'

			# Shrink the size of selected exemplars to be the one defined by exemplar_size
			# This size may not be multiples of batch_size
			exemplar_data[dtype] = exemplar_data[dtype][: exemplar_size[dtype]]
			distillations[dtype] = distillations[dtype][: exemplar_size[dtype]] if 'distillation' in experiment_type else None
			exemplars[dtype] = Exemplars(exemplar_data[dtype], distillations[dtype])

			print(f"Obtain {exemplars[dtype].size()} exemplars for {dtype} data", file = sys.stderr)
		
		""" Truncate distillation ids if needed """
		for dtype in ["train", "valid"]:
			selected_ids[dtype] = selected_ids[dtype][: exemplar_size[dtype]]

		if not return_selected_ids:
			return exemplars
		else:
			return exemplars, selected_ids


def compute_distillation(model, task, selected_ids):
	"""
	This function compute the distillation for selected ids of given task
	as last hidden of all lstm
	@param model (LM_deep)
	@param task (Task)
	@param selected_ids ( dict(str: List(int)) )
	@return dict(str: torch.Tensor): distillation
	"""

	distillations = dict()

	model.eval()
	for dtype in ["train", "valid"]:	
			
		distillation_data = None
		n_batch = len(selected_ids[dtype]) // task.batch_size
		distillation_list = []

		#  Go through each batch to compute distillation
		for i in range(n_batch):
			
			input_var, label_var, feats_var, lengths,\
			refs, featStrs, sv_indexes, _, _, _, _= task.next_batch(data_type = dtype,
														selected_indices = selected_ids[dtype][i * task.batch_size:\
																(i + 1) * task.batch_size]) 		
			with torch.no_grad():
				if model.model_type == "lm":
					_, last_hidden = model(input_var, task, feats_var, keep_last=False)
				else:
					target_var = input_var.clone()
					model.set_prior(False)
					_, last_hidden = model(input_var, input_lengths = lengths, target_seq = target_var, target_lengths = lengths, conds_seq = feats_var, dataset = task)

				output_word_proba = model.output_prob
			print(f"Output word probability has shape {output_word_proba.size()}", file = sys.stderr)

			distillation_list.append(output_word_proba)
		
		# Notice that the distillation here is unnormalized and
		# to pass through softmax to obtain real distribution
		distillation_data = torch.cat(distillation_list)
		print(f"Distillation data has shape {distillation_data.size()}", file = sys.stderr)

		distillations[dtype] = distillation_data

	return distillations
