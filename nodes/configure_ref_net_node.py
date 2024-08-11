from ..modules.ref_block import inject_ref_blocks


class ConfigureRefNetNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "reference"

    def apply(self, model):
        inject_ref_blocks(model.model.diffusion_model)
        return (model,)
