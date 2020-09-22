import sys
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

def hotfix_pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
	
    lengths = torch.as_tensor(lengths, dtype=torch.int64)
    lengths = lengths.cpu()
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    data, batch_sizes = \
        torch._C._VariableFunctions._pack_padded_sequence(input, lengths, batch_first)
    return PackedSequence(data, batch_sizes, sorted_indices)


class Encoder(nn.Module):
	"""
	Encoder of sentences
	"""
	def __init__(self, vocab_size, hidden_size, n_layers=1, dropout=0):
		super(Encoder, self).__init__()
		
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.rnn = nn.LSTM(vocab_size, hidden_size, n_layers, dropout=dropout, bidirectional=True, batch_first=True)
		
	def forward(self, input_seqs, input_lengths):
		'''
		@parma input_seqs: (batch_size, max_len)
		@param input_lengths: (max_len,)
		'''
		# Note: we run this all at once (over multiple batches of multiple sequences)

		# Pack padded sentences
		packed = hotfix_pack_padded_sequence(input_seqs, input_lengths, batch_first=True, enforce_sorted = False) # same size

		# Encode by passing into rnn
		outputs, (hidden, cell) = self.rnn(packed)

		# Pad output
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True) # unpack (back to padded)

		outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs

		return outputs, hidden

