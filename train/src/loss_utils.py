import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from copy import deepcopy

SD14_TO_SD21_RATIO = 1.5

# get token index in text
def get_word_idx(text: str, tgt_word, tokenizer):

    tgt_word = tgt_word.lower()

    # ignore the first and last token
    encoded_text = tokenizer.encode(text)[1:-1]
    encoded_tgt_word = tokenizer.encode(tgt_word)[1:-1]

    # find the idx of target word in text
    first_token_idx = -1
    for i in range(len(encoded_text)):
        if encoded_text[i] == encoded_tgt_word[0]:

            if len(encoded_text) > 0:
                # check the following 
                following_match = True
                for j in range(1, len(encoded_tgt_word)):
                    if encoded_text[i + j] != encoded_tgt_word[j]:
                        following_match = False
                if not following_match:
                    continue
            # for a single encoded idx, just take it
            first_token_idx = i

            break

    assert first_token_idx != -1, "word not in text"

    # add 1 for sot token
    tgt_word_tokens_idx_ls = [i + 1 + first_token_idx for i in range(len(encoded_tgt_word))]

    # sanity check
    encoded_text = tokenizer.encode(text)

    decoded_token_ls = []

    for word_idx in tgt_word_tokens_idx_ls:
        text_decode = tokenizer.decode([encoded_text[word_idx]]).strip("#")
        decoded_token_ls.append(text_decode)

    decoded_tgt_word = "".join(decoded_token_ls)
    
    tgt_word_ls = tgt_word.split(" ")
    striped_tgt_word = "".join(tgt_word_ls).strip("#")

    assert decoded_tgt_word == striped_tgt_word, "decode_text != striped_tar_wd"

    return tgt_word_tokens_idx_ls

# get attn loss by resolution
def get_grounding_loss_by_layer(_gt_seg_list, word_token_idx_ls, res,
                                input_attn_map_ls, is_training_sd21):
    if is_training_sd21:
        # training with sd21, using resolution 768 = 512 * 1.5
        res = int(SD14_TO_SD21_RATIO * res)

    gt_seg_list = deepcopy(_gt_seg_list)

    # reszie gt seg map to the same size with attn map
    resize_transform = transforms.Resize((res, res))

    for i in range(len(gt_seg_list)):
        gt_seg_list[i] = resize_transform(gt_seg_list[i])
        gt_seg_list[i] = gt_seg_list[i].squeeze(0) # 1, 1, res, res => 1, res, res
        # add binary
        gt_seg_list[i] = (gt_seg_list[i] > 0.0).float()


    ################### token loss start ###################
    # Following code is adapted from
    # https://github.com/silent-chen/layout-guidance/blob/08b687470f911c7f57937012bdf55194836d693e/utils.py#L27
    token_loss = 0.0
    for attn_map in input_attn_map_ls:
        b, H, W, j = attn_map.shape
        for i in range(len(word_token_idx_ls)): # [[word1 token_idx1, word1 token_idx2, ...], [word2 token_idx1, word2 token_idx2, ...]]
            obj_loss = 0.0
            single_word_idx_ls = word_token_idx_ls[i] #[token_idx1, token_idx2, ...]
            mask = gt_seg_list[i]

            for obj_position in single_word_idx_ls:
                # ca map obj shape 8 * 16 * 16
                ca_map_obj = attn_map[:, :, :, obj_position].reshape(b, H, W)

                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)

                obj_loss += (1.0 - torch.mean(activation_value)) ** 2

            token_loss += (obj_loss/len(single_word_idx_ls))

    # normalize with len words
    token_loss = token_loss / len(word_token_idx_ls)
    ################## token loss end ##########################

    ################## pixel loss start ######################
    # average cross attention map on different layers
    avg_attn_map_ls = []
    for i in range(len(input_attn_map_ls)):
        avg_attn_map_ls.append(
            input_attn_map_ls[i].reshape(-1, res, res, input_attn_map_ls[i].shape[-1]).mean(0)
        )
    avg_attn_map = torch.stack(avg_attn_map_ls, dim=0)
    avg_attn_map = avg_attn_map.sum(0) / avg_attn_map.shape[0]
    avg_attn_map = avg_attn_map.unsqueeze(0)

    bce_loss_func = nn.BCELoss()
    pixel_loss = 0.0

    for i in range(len(word_token_idx_ls)):
        word_cross_attn_ls = []
        for token_idx in word_token_idx_ls[i]:
            word_cross_attn_ls.append(
                avg_attn_map[..., token_idx]
            )
        word_cross_attn_ls = torch.stack(word_cross_attn_ls, dim=0).sum(dim=0)
        pixel_loss += bce_loss_func(word_cross_attn_ls, gt_seg_list[i])

    # average with len word_token_idx_ls
    pixel_loss = pixel_loss / len(word_token_idx_ls)
    ################## pixel loss end #########################

    return {
        "token_loss" : token_loss,
        "pixel_loss": pixel_loss,
    }


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

