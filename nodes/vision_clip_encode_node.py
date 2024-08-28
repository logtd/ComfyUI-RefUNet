
import os

import torch

import comfy.model_management
import comfy.utils
from comfy.clip_vision import clip_preprocess

from .. import REPO_DIR


class VisionClipEncodeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_vision": ("CLIP_VISION", ),
            "clip_image": ("IMAGE",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "clip"

    def encode(self, clip_vision, clip_image, strength):
        dtype=clip_vision.dtype
        device=comfy.model_management.get_torch_device()
        clip_image = clip_preprocess(clip_image.clone(), 224)
        clip_embeds = clip_vision.encode_image(clip_image.permute(0, 2, 3, 1))["last_hidden_state"].to(dtype).to(device)
        clip_embeds = clip_embeds * strength
        clip_fc_path = os.path.join(REPO_DIR, "models","clip_fc.safetensors")
        sd = comfy.utils.load_torch_file(clip_fc_path)
        self.clip_fc = torch.nn.Linear(1024, 768, bias=True).to(clip_embeds.dtype).to(clip_embeds.device)
        self.clip_fc.load_state_dict(sd)

        clip_in = clip_embeds
        clip_out = self.clip_fc(clip_in) * strength
        clip_out = clip_out.to('cpu')
           
        return ([[clip_out, {"pooled_output": clip_out}]], )
