import torch
import torch.nn.functional as F

SD14_TO_SD21_RATIO = 1.5

# The original TokenCompose token/pixel grounding utilities are retained below as
# commented reference. They are not invoked by the boundary-consistency training
# pipeline.
"""
def get_word_idx(text: str, tgt_word, tokenizer):
    ...


def get_grounding_loss_by_layer(_gt_seg_list, word_token_idx_ls, res,
                                input_attn_map_ls, is_training_sd21):
    ...
"""


def _gaussian_blur(tensor, kernel_size, sigma):
    if kernel_size <= 0 or sigma <= 0:
        return tensor

    # force odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    coords = torch.arange(kernel_size, dtype=tensor.dtype, device=tensor.device)
    coords = coords - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.matmul(kernel_1d.unsqueeze(1), kernel_1d.unsqueeze(0))
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    padding = kernel_size // 2
    return F.conv2d(tensor, kernel_2d, padding=padding)


def get_boundary_consistency_loss(
    boundary_map,
    token_mask,
    res,
    input_attn_map_ls,
    is_training_sd21,
    blur_kernel_size=0,
    blur_sigma=0.0,
):
    if len(input_attn_map_ls) == 0:
        zero = boundary_map.new_zeros(())
        return zero, zero

    if boundary_map.dim() != 4:
        raise ValueError("boundary_map is expected to have shape [B, 1, H, W]")

    # normalize to float for safe ops
    boundary_map = boundary_map.to(dtype=torch.float32)

    if token_mask.dim() == 2:
        token_mask = token_mask[0]
    token_mask = token_mask.to(boundary_map.device, dtype=boundary_map.dtype)

    attn_token_count = input_attn_map_ls[0].shape[-1]
    token_mask = token_mask[:attn_token_count]
    valid_token_count = token_mask.sum().clamp(min=1.0)

    total_boundary = boundary_map.new_zeros(())
    total_smooth = boundary_map.new_zeros(())
    map_counter = 0

    for attn_map in input_attn_map_ls:
        _, attn_H, attn_W, tokens = attn_map.shape
        target_res = int(SD14_TO_SD21_RATIO * res) if is_training_sd21 else res
        if attn_H != target_res or attn_W != target_res:
            target_res = (attn_H, attn_W)
        else:
            target_res = (target_res, target_res)

        resized_boundary = F.interpolate(boundary_map, size=target_res, mode="bilinear", align_corners=False)
        resized_boundary = _gaussian_blur(resized_boundary, blur_kernel_size, blur_sigma)
        resized_boundary = resized_boundary.squeeze(1)
        resized_boundary = resized_boundary.clamp(min=0.0)

        if resized_boundary.max() > 0:
            resized_boundary = resized_boundary / resized_boundary.max()

        attn_map = attn_map.reshape(-1, attn_H, attn_W, tokens)
        mean_attn = attn_map.mean(dim=0).permute(2, 0, 1)  # tokens, H, W

        norm_boundary = resized_boundary / resized_boundary.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)

        per_token_boundary = (mean_attn * norm_boundary.unsqueeze(0)).sum(dim=(1, 2))
        total_boundary = total_boundary + (per_token_boundary * token_mask[:tokens]).sum()

        grad_x = mean_attn[:, :, 1:] - mean_attn[:, :, :-1]
        grad_y = mean_attn[:, 1:, :] - mean_attn[:, :-1, :]
        grad_energy = grad_x.pow(2).mean(dim=(1, 2)) + grad_y.pow(2).mean(dim=(1, 2))
        total_smooth = total_smooth + (grad_energy * token_mask[:tokens]).sum()

        map_counter += 1

    if map_counter == 0:
        zero = boundary_map.new_zeros(())
        return zero, zero

    total_boundary = total_boundary / (valid_token_count * map_counter)
    total_smooth = total_smooth / (valid_token_count * map_counter)

    return total_boundary, total_smooth


def _gaussian_blur(tensor, kernel_size, sigma):
    if kernel_size <= 0 or sigma <= 0:
        return tensor

    # force odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    coords = torch.arange(kernel_size, dtype=tensor.dtype, device=tensor.device)
    coords = coords - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.matmul(kernel_1d.unsqueeze(1), kernel_1d.unsqueeze(0))
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    padding = kernel_size // 2
    return F.conv2d(tensor, kernel_2d, padding=padding)


def get_boundary_consistency_loss(
    boundary_map,
    token_mask,
    res,
    input_attn_map_ls,
    is_training_sd21,
    blur_kernel_size=0,
    blur_sigma=0.0,
):
    if len(input_attn_map_ls) == 0:
        zero = boundary_map.new_zeros(())
        return zero, zero

    if boundary_map.dim() != 4:
        raise ValueError("boundary_map is expected to have shape [B, 1, H, W]")

    # normalize to float for safe ops
    boundary_map = boundary_map.to(dtype=torch.float32)

    if token_mask.dim() == 2:
        token_mask = token_mask[0]
    token_mask = token_mask.to(boundary_map.device, dtype=boundary_map.dtype)

    attn_token_count = input_attn_map_ls[0].shape[-1]
    token_mask = token_mask[:attn_token_count]
    valid_token_count = token_mask.sum().clamp(min=1.0)

    total_boundary = boundary_map.new_zeros(())
    total_smooth = boundary_map.new_zeros(())
    map_counter = 0

    for attn_map in input_attn_map_ls:
        _, attn_H, attn_W, tokens = attn_map.shape
        target_res = int(SD14_TO_SD21_RATIO * res) if is_training_sd21 else res
        if attn_H != target_res or attn_W != target_res:
            target_res = (attn_H, attn_W)
        else:
            target_res = (target_res, target_res)

        resized_boundary = F.interpolate(boundary_map, size=target_res, mode="bilinear", align_corners=False)
        resized_boundary = _gaussian_blur(resized_boundary, blur_kernel_size, blur_sigma)
        resized_boundary = resized_boundary.squeeze(1)
        resized_boundary = resized_boundary.clamp(min=0.0)

        if resized_boundary.max() > 0:
            resized_boundary = resized_boundary / resized_boundary.max()

        attn_map = attn_map.reshape(-1, attn_H, attn_W, tokens)
        mean_attn = attn_map.mean(dim=0).permute(2, 0, 1)  # tokens, H, W

        norm_boundary = resized_boundary / resized_boundary.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)

        per_token_boundary = (mean_attn * norm_boundary.unsqueeze(0)).sum(dim=(1, 2))
        total_boundary = total_boundary + (per_token_boundary * token_mask[:tokens]).sum()

        grad_x = mean_attn[:, :, 1:] - mean_attn[:, :, :-1]
        grad_y = mean_attn[:, 1:, :] - mean_attn[:, :-1, :]
        grad_energy = grad_x.pow(2).mean(dim=(1, 2)) + grad_y.pow(2).mean(dim=(1, 2))
        total_smooth = total_smooth + (grad_energy * token_mask[:tokens]).sum()

        map_counter += 1

    if map_counter == 0:
        zero = boundary_map.new_zeros(())
        return zero, zero

    total_boundary = total_boundary / (valid_token_count * map_counter)
    total_smooth = total_smooth / (valid_token_count * map_counter)

    return total_boundary, total_smooth

