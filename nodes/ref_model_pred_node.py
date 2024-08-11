import comfy.sd
import comfy.model_sampling
import comfy.latent_formats


class X0Ref(comfy.model_sampling.EPS):
    def calculate_input(self, sigma, noise):
        return noise
    def calculate_denoised(self, sigma, model_output, model_input):
        return model_output


class RefModelSamplingPredNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "reference"

    def patch(self, model):
        m = model.clone()

        sampling_base = comfy.model_sampling.ModelSamplingDiscrete
        sampling_type = X0Ref

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)

        m.add_object_patch("model_sampling", model_sampling)
        return (m, )
