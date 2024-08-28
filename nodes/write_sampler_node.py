import torch
import comfy.samplers
from comfy.samplers import KSAMPLER

from ..utils.sampler_utils import get_sampler_fn, create_sampler
from ..utils.ref_constants import SD1_REF_MAP


@torch.no_grad()
def sample_write(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    return model(x, sigmas[0] * s_in, **extra_args)


class WriteSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "sampler_name": (["REFERENCE_WRITE"] +  comfy.samplers.SAMPLER_NAMES, ),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "ref_bank": ("REF_BANK",),
            "mask_dilation": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
        }, "optional": {
            "sampler": ("SAMPLER",),
            "opt_attn_map": ("ATTN_MAP",),
            "masks": ("MASK",),
        }}
    RETURN_TYPES = ("SAMPLER","SIGMAS")
    FUNCTION = "build"

    CATEGORY = "reference/sampling"

    def build(self, sampler_name, start_percent, end_percent, ref_bank, mask_dilation, sampler=None, opt_attn_map=SD1_REF_MAP, masks=None):
        if sampler_name == 'REFERENCE_WRITE':
            sampler_fn = sample_write
        else:
            sampler_fn = get_sampler_fn(sampler_name)
        sampler_fn = create_sampler(sampler_fn, ref_bank, opt_attn_map, 'WRITE', start_percent, end_percent, mask_dilation, masks)
        
        if sampler is None:
            sampler = KSAMPLER(sampler_fn)
        else:
            sampler.sampler_function = sampler_fn

        return (sampler, torch.Tensor([0]))
