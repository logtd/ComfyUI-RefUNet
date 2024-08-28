import torch

import comfy.k_diffusion.sampling as k_diffusion_sampling


def get_stepper(model, model_options, sigmas, start_percent, end_percent):
    prev_step = [0] # hack for special samplers
    def sample_step(x, sigma, **extra_args):
        step = torch.where(sigma[0] == sigmas)[0]
        if not len(step):
            step = prev_step[0]
        prev_step[0] = step
        step_percent = step.item() / model_options['transformer_options']['TOTAL_STEPS']
        ref_on = start_percent <= step_percent <= end_percent
        model_options['transformer_options']['REF_ON'] = ref_on

        output =  model(x, sigma, **extra_args)

        del model_options['transformer_options']['REF_ON']

        return output
    
    return sample_step


def create_sampler(sample_fn, ref_bank, ref_map, ref_type, start_percent=0, end_percent=1, mask_dilation=0, masks=None):
    @torch.no_grad()
    def sample(model, latents, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
        model_options = extra_args.get('model_options', {})
        transformer_options = model_options.get('transformer_options', {})

        if ref_type == 'WRITE':
            ref_bank.clear()
            for block_idx in ref_map:
                ref_bank[block_idx] = {}
            
        model_options = {
            **model_options,
            'transformer_options': {
                **transformer_options,
                'REF_TYPE': ref_type,
                'REF_BANK': ref_bank,
                'TOTAL_STEPS': len(sigmas),
                'REF_MASK': masks,
                'REF_MASK_DILATION': mask_dilation,
            }
        }
        extra_args = {**extra_args, 'model_options': model_options}

        sampler_stepper = get_stepper(model, model_options, sigmas, start_percent, end_percent)

        output = sample_fn(sampler_stepper, latents, sigmas, extra_args=extra_args, callback=callback, disable=disable, **extra_options)

        if 'REF_BANK' in model_options['transformer_options']:
            del model_options['transformer_options']['REF_BANK']

        if 'REF_TYPE' in model_options['transformer_options']:
            del model_options['transformer_options']['REF_TYPE']

        if 'TOTAL_STEPS' in model_options['transformer_options']:
            del model_options['transformer_options']['TOTAL_STEPS']

        return output
    
    return sample


def get_sampler_fn(sampler_name):
    if sampler_name == "dpm_fast":
        def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            total_steps = len(sigmas) - 1
            return k_diffusion_sampling.sample_dpm_fast(model, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_fast_function
    elif sampler_name == "dpm_adaptive":
        def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable):
            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return k_diffusion_sampling.sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_adaptive_function
    else:
        sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))
    return sampler_function
