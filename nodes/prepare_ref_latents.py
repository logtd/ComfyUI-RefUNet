

class PrepareRefLatentsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "src_latents": ("LATENT",),
            "ref_latents": ("LATENT",),
        }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"

    CATEGORY = "reference"

    def apply(self, src_latents, ref_latents):
        # This is mostly a trick node to ensure that comfy executes the sampling in the correct order
        return (src_latents,)
