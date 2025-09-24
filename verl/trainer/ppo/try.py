import math
import torch
import numpy as np
loss_mat = torch.tensor([[1,2,3,4,5], [5.0,4,3,2,1],[1,2,3,4,1]])
loss_mask = torch.tensor([[1,1,1,0,0], [1,1,1,0,0], [1,1,1,0,0]])

lambda_ours = 0
token_len_for_each_sample = loss_mask.sum(dim=-1).float()
mu = torch.mean(token_len_for_each_sample)
sigma = torch.std(token_len_for_each_sample)
sigma = sigma.clamp_min(1e-8)
reducer = 1/30
normalized = (token_len_for_each_sample - mu)/sigma * reducer + 1
new_loss_mat = loss_mat * loss_mask
print(new_loss_mat)
# lambda_ours = torch.clamp(lambda_ours, -3.0, 3.0)
weighted = normalized.pow(lambda_ours)
print(f"normalized:{normalized}")
print(f"weighted:{weighted}")
sample_size = loss_mask.shape[0]
f = torch.softmax(weighted, dim=0) * sample_size # torch.tensor([1., 1., 1.])
final_loss_for_each_token = f * new_loss_mat.sum(dim=-1)
print(final_loss_for_each_token)
sum_of_token_len_of_all_samples = torch.sum(token_len_for_each_sample)
loss = torch.sum(final_loss_for_each_token) /sum_of_token_len_of_all_samples

print(loss)

# print(torch.sum(loss_mask, dim=-1))
# print(torch.sum(loss_mask))
# print(torch.sum(loss_mat * loss_mask, dim=-1))

# seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
# loss = torch.sum(seq_losses)
# def masked_sum(values, mask, axis=None):
#     """Compute mean of tensor with a masked values."""
#     # If NaNs exist out of mask, replace NaNs in values with a value that
#     # won't affect the sum (e.g., 0 for masked regions)
#     valid_values = torch.where(mask.bool(), values, 0.0)
#     print(valid_values)
#     print((valid_values * mask)) #和上面一样呀
#     return (valid_values * mask).sum(axis=axis)


# def masked_mean(values, mask, axis=None):
#     """
#     Compute the mean of `values` over elements selected by `mask`.

#     Args:
#         values (Tensor): Input tensor.
#         mask (Tensor): Boolean or numeric mask of the same shape as `values`.
#         axis (int or tuple of int, optional): Dimension(s) along which to compute the mean.
#             Defaults to None (over all elements).

#     Returns:
#         Tensor: Masked mean, with shape equal to `values` reduced over `axis`.
#     """
#     s = masked_sum(values, mask, axis) # 30
#     return s / (mask.sum(axis=axis) + 1e-8) # 30/10




# # loss = masked_mean(loss_mat, loss_mask)
# # print(loss_mask.sum())
# print(loss_mask.shape) # torch.Size([3, 5])


# # weighted3-reducer-1-over-30
# token_len_for_each_sample = loss_mask.sum(dim=-1).float() # tensor([2, 3, 5])
# mu = torch.mean(token_len_for_each_sample)
# sigma = torch.std(token_len_for_each_sample)
# reducer = 1/3
# lamda = 2
# normalized = (token_len_for_each_sample - mu)/sigma * reducer + 1 # tensor([0.9709, 0.9927, 1.0364])
# print(normalized)
# clipped_normalized = torch.clip(normalized, min=0.1)
# print(clipped_tensor)

# weighted = normalized.pow(lamda)

# new_loss_mat = loss_mat * loss_mask
# normalized_loss_for_each_token = normalized * new_loss_mat.sum(dim=-1)
# weighted_loss_for_each_token = weighted * new_loss_mat.sum(dim=-1)
# sample_size = loss_mask.shape[0]

# f = torch.softmax(weighted, dim=0) * sample_size
# print(f)
# print(sample_size)
# final_loss_for_each_token = f * new_loss_mat.sum(dim=-1)
# # sum_of_loss_of_all_samples = masked_sum(loss_mat, loss_mask)
# # print(sum_of_loss_of_all_samples)
# sum_of_token_len_of_all_samples = torch.sum(token_len_for_each_sample) 
# loss_weighted3 = torch.sum(normalized_loss_for_each_token) /sum_of_token_len_of_all_samples
# loss_lambda_grpo = torch.sum(final_loss_for_each_token) /sum_of_token_len_of_all_samples
# print(loss_weighted3)
# print(loss_lambda_grpo)






