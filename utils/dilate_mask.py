import math
import torch
import torch.nn.functional as F


def dilate_mask(n: torch.Tensor, transformer_options):
    mask = transformer_options.get('REF_MASK', None)
    mask_dilation = transformer_options.get('REF_MASK_DILATION', 0)

    if mask is None or mask_dilation <= 1:
        return n.clone()

    H, W = mask.shape[-2:]
    scale = 1 << int(
        math.ceil(math.log2((H * W) / n.shape[-2]) / 2)
    )
    H, W = math.ceil(H / scale), math.ceil(W / scale)
    resized_mask = F.interpolate(mask.unsqueeze(1), (H, W)).to(n.dtype).to(n.device)
    dilation_kernel = torch.ones(
        (1, 1, mask_dilation, mask_dilation), dtype=n.dtype, device=n.device
    )
    dilated_mask = (
        F.conv2d(resized_mask, dilation_kernel, padding=1)
        .view(-1, H * W, 1)
        .clamp(0.0, 1.0)
    )
    n = n * dilated_mask

    return n.clone()