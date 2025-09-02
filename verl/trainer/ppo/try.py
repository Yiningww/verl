import math
import torch
loss_mat = torch.tensor([[1.0,2.0,3.0,4.0,5.0], [5.0,4,3,2,1],[1.0,2,3,4,5]])
loss_mask = torch.tensor([[1.0,1,0,0,0], [1.0,1,1,0,0], [1.0,1,1,1,1]])
# print(torch.sum(loss_mask, dim=-1))
# print(torch.sum(loss_mask))
# print(torch.sum(loss_mat * loss_mask, dim=-1))

seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
loss = torch.sum(seq_losses)
def masked_sum(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    # If NaNs exist out of mask, replace NaNs in values with a value that
    # won't affect the sum (e.g., 0 for masked regions)
    valid_values = torch.where(mask.bool(), values, 0.0)
    print(valid_values)
    print((valid_values * mask)) #和上面一样呀
    return (valid_values * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None):
    """
    Compute the mean of `values` over elements selected by `mask`.

    Args:
        values (Tensor): Input tensor.
        mask (Tensor): Boolean or numeric mask of the same shape as `values`.
        axis (int or tuple of int, optional): Dimension(s) along which to compute the mean.
            Defaults to None (over all elements).

    Returns:
        Tensor: Masked mean, with shape equal to `values` reduced over `axis`.
    """
    s = masked_sum(values, mask, axis) # 30
    return s / (mask.sum(axis=axis) + 1e-8) # 30/10




# loss = masked_mean(loss_mat, loss_mask)
# print(loss_mask.sum())
print(loss_mask.shape) # torch.Size([3, 5])


# yining-weighted3-reducer-1-over-30
token_len_for_each_sample = loss_mask.sum(dim=-1) # tensor([2, 3, 5])
mu = torch.mean(token_len_for_each_sample)
print(mu)
sigma = torch.std(token_len_for_each_sample)
print(sigma)
reducer = 1/30
normalized = (token_len_for_each_sample - mu)/sigma * reducer + 1 # tensor([0.9709, 0.9927, 1.0364])

new_loss_mat = loss_mat * loss_mask
normalized_loss_for_each_token = normalized * new_loss_mat.sum(dim=-1)

# sum_of_loss_of_all_samples = masked_sum(loss_mat, loss_mask)
# print(sum_of_loss_of_all_samples)
sum_of_token_len_of_all_samples = torch.sum(token_len_for_each_sample) 
loss = torch.sum(normalized_loss_for_each_token) /sum_of_token_len_of_all_samples
print(loss)
# loss2 = masked_mean(loss_mat, loss_mask)
# print(loss2)





