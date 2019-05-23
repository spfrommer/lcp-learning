import torch

# Turns a n-dim vector into an nx2 matrix
# If pos_first, first column is all positive values (negs set to 0)
# Second col is abs value of all negative values (pos set to 0)
def split_sign(tensor, pos_first):
    tensor_pos = tensor.clone()
    tensor_neg = tensor.clone()

    tensor_pos[tensor_pos < 0] = 0
    tensor_neg[tensor_neg > 0] = 0

    tensor_pos = tensor_pos.unsqueeze(1)
    tensor_neg = tensor_neg.unsqueeze(1) * (-1)
    if pos_first:
        tensor = torch.cat((tensor_pos, tensor_neg), 1)
    else:
        tensor = torch.cat((tensor_neg, tensor_pos), 1)
    return tensor

# Inverse of split_sign
def join_sign(tensor, pos_first):
    tensor = tensor.clone()
    if pos_first:
        tensor[:, 1] = tensor[:, 1] * -1
    else:
        tensor[:, 0] = tensor[:, 0] * -1
    
    return tensor.sum(1)
