import torch
from torch.nn import functional as F
from torch.autograd import Variable
import sys

def sequence_mask(sequence_length, max_len = None):
    """
    Compute sequence mask
    """

    if max_len is None:
        max_len = sequence_length.data.max()

    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)

    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()

    seq_length_expand = (sequence_length.unsqueeze(1)
                            .expand_as(seq_range_expand))

    return seq_range_expand < seq_length_expand

def masked_kl_divergence(logits, target, length, T, old_indices_mask = None):
    """
    This func calcualte the average kl-divergence \n
    @param logits: unnormalized probability for each word with 
                    shape (batch_size, max_len, vocab_size)
                    given by new model\n
    @param target: unnormalized probability for each word with 
                    shape (batch_size, max_len, vocab_size)
                    given by models trained on previous tasks \n
    @param length: length of each sentence \n
    @param T: temperature in the regular softmax \n
    @return: loss: average loss masked by the length \n
    """

    # Flatten the logits -> (batch * max_len, vocab_size)
    logits_flat = logits.view(-1, logits.size(-1))

    # Flatten the target
    target_flat = target.view(-1, target.size(-1))

    # Log_softmax the logits
    log_probs_flat = F.log_softmax(logits_flat / T, dim = -1)

    # Softmax the target
    target_probs = F.softmax(target_flat / T, dim = -1)

    # Calculate KL-divergence
    losses_flat = F.kl_div(input = log_probs_flat, target = target_probs,\
                            reduction = "none").sum(dim = 1)

    # Reshape to (batch_size, max_len)
    losses = losses_flat.view(*logits.size()[: -1])

    # Mask the loss by sequence length
    mask = sequence_mask(sequence_length = length, max_len = target.size(1))

    if old_indices_mask != None:
        # mask the loss by old_indices
        mask = mask * old_indices_mask

    losses = losses * mask.float()

    # Calculate per word loss
    if mask.float().sum() > 0:
        loss = losses.sum() / mask.float().sum()
    else:
        loss = torch.tensor(0)

    return loss


