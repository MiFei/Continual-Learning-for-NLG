import sys
import torch
from torch.autograd import Variable
USE_CUDA = True

torch.manual_seed(1111)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1111)
torch.backends.cudnn.deterministic = True

def sample_gaussian(mu, logvar, std):
    """
    Sample point from gaussian
    @param mu(torch.Tensor): mean of gaussian
    @param logvar(torch.Tensor): log of variance of gaussian
    @param std(float): std of normal distribution with zero mean to generate the weight of offset specified by logvar
    @return point(troch.Tensor): sampled point
    """
    
    assert mu.size() == logvar.size()
    _size = logvar.size()
    epsilon = Variable(torch.normal(mean=torch.zeros(*_size), std=std))
    g_std = torch.exp(0.5 * logvar)
    if USE_CUDA:
            epsilon = epsilon.cuda()
    return mu + g_std * epsilon