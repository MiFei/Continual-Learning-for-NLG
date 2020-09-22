import torch
from torch.nn import functional
from torch.autograd import Variable

def sequence_mask(sequence_length, max_len=None):
	if max_len is None:
		max_len = sequence_length.data.max()
	batch_size = sequence_length.size(0)
	seq_range = torch.arange(0, max_len).long() # andy
	seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
	seq_range_expand = Variable(seq_range_expand)
	if sequence_length.is_cuda:
		seq_range_expand = seq_range_expand.cuda()
	seq_length_expand = (sequence_length.unsqueeze(1)
						 .expand_as(seq_range_expand))
	return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length, return_vec = False):
	"""
	@param logits: A Variable containing a FloatTensor of size
			(batch, max_len, num_classes) which contains the
			unnormalized probability for each class. \n 
	@param target: A Variable containing a LongTensor of size
			(batch, max_len) which contains the index of the true
			class for each corresponding step. \n
	@param length: A Variable containing a LongTensor of size (batch,)
			which contains the length of each data in a batch. \n
	@param return_vec: A boolean defining whether to return loss vector containing loss for each sentence \n
	@return loss: An average loss value masked by the length. \n
	""" 

	# logits_flat: (batch * max_len, num_classes)
	logits_flat = logits.view(-1, logits.size(-1))
	# log_probs_flat: (batch * max_len, num_classes)
	log_probs_flat = functional.log_softmax(logits_flat, dim=1)
	# target_flat: (batch * max_len, 1)
	target_flat = target.view(-1, 1)
	# losses_flat: (batch * max_len, 1)
	losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
	# losses: (batch, max_len)
	losses = losses_flat.view(*target.size())
	# mask: (batch, max_len)
	mask = sequence_mask(sequence_length=length, max_len=target.size(1))

	losses = losses * mask.float()
	if not return_vec:
		loss = losses.sum() / length.float().sum() # per word loss
	else:
		loss = losses.sum(dim = 1) / length.float()
		print(f"The loss has shape {loss.shape}")

	return loss
