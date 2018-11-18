import torch
from torch.autograd import Variable

def to_var(x):
    """Converts numpy to variable."""
    x = torch.from_numpy(x).float()
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()
