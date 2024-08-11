import comfy.samplers
from comfy.samplers import KSAMPLER

from ..utils.sampler_utils import get_sampler_fn, create_sampler
from ..utils.ref_constants import SD1_REF_MAP



class ReadSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "sampler_name": (comfy.samplers.SAMPLER_NAMES, ),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "ref_bank": ("REF_BANK",)
        }, "optional": {
            "sampler": ("SAMPLER",),
            # "opt_attn_map": ("ATTN_MAP",),
        }}
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"

    CATEGORY = "reference/sampling"

    def build(self, sampler_name, start_percent, end_percent, ref_bank, sampler=None, opt_attn_map=SD1_REF_MAP):
        sampler_fn = get_sampler_fn(sampler_name)
        sampler_fn = create_sampler(sampler_fn, ref_bank, opt_attn_map, 'READ', start_percent, end_percent)
        
        if sampler is None:
            sampler = KSAMPLER(sampler_fn)
        else:
            sampler.sampler_function = sampler_fn

        return (sampler, )