import torch
from comfy.ldm.modules.attention import BasicTransformerBlock
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel

from ..utils.module_utils import isinstance_str


class RefTransformerBlock(BasicTransformerBlock):
    def configure(self, block, idx):
        self.block = block
        self.idx = idx
        self.block_idx = (block, idx)

    def forward(self, x, context=None, transformer_options={}):
        extra_options = {}
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches = {}
        transformer_patches_replace = {}

        for k in transformer_options:
            if k == "patches":
                transformer_patches = transformer_options[k]
            elif k == "patches_replace":
                transformer_patches_replace = transformer_options[k]
            else:
                extra_options[k] = transformer_options[k]

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head
        extra_options["attn_precision"] = self.attn_precision

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        orig_normal = self.norm1(x)
        n = orig_normal.clone()

        conds = transformer_options['cond_or_uncond']
        len_conds = len(conds)
        n_frames = len(x) // len_conds

        ref_type = transformer_options.get('REF_TYPE', None)
        ref_bank = transformer_options.get('REF_BANK', None)
        ref_on = transformer_options.get('REF_ON', False)
        REF_UNCOND_READ = False

        context_attn1 = n

        if ref_type == 'WRITE' and ref_bank is not None and self.block_idx in ref_bank and ref_on:
            for idx, cond in enumerate(conds):
                ref_bank[self.block_idx][cond] = n[idx*n_frames:(idx+1)*n_frames].clone().cpu()
        elif ref_type == 'READ' and ref_bank is not None and self.block_idx in ref_bank and ref_on:
            ref_n = []
            for idx, cond in enumerate(conds):
                if cond in ref_bank[self.block_idx]:
                    ref_n.append(ref_bank[self.block_idx][cond].to(x.device).repeat(n_frames, 1, 1))
                    if cond == 1:
                        REF_UNCOND_READ = True
                else:
                    ref_n.append(context_attn1[idx*n_frames:(idx+1)*n_frames]) # TODO make this faster
            ref_n = torch.cat(ref_n)
            context_attn1 = torch.cat([context_attn1, ref_n], dim=1)

        value_attn1 = context_attn1

        if "attn1_patch" in transformer_patches:
            patch = transformer_patches["attn1_patch"]
            for p in patch:
                n, context_attn1, value_attn1 = p(n, context_attn1, value_attn1, extra_options)

        if block is not None:
            transformer_block = (block[0], block[1], block_index)
        else:
            transformer_block = None
        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block

        if block_attn1 in attn1_replace_patch:
            q = self.attn1.to_q(n)
            context_attn1 = self.attn1.to_k(context_attn1)
            value_attn1 = self.attn1.to_v(value_attn1)
            hidden_states = attn1_replace_patch[block_attn1](q, context_attn1, value_attn1, extra_options)
            hidden_states = self.attn1.to_out(hidden_states)
            del q
        else:
            if hasattr(self.attn1, 'veevee'):
                hidden_states = self.attn1(n, context_attn1, value_attn1, extra_options=extra_options)
            else:
                hidden_states = self.attn1(n, context_attn1, value_attn1)
        
        if REF_UNCOND_READ:
            hidden_states_uc_c = hidden_states.clone()
            uc_mask = []
            for cond in conds:
                uc_mask.append(torch.Tensor([cond] * n_frames))
            uc_mask = torch.cat(uc_mask).to(hidden_states.device).bool()
            hidden_states_uc_c[uc_mask] = self.attn1(orig_normal[uc_mask]) 
            hidden_states = hidden_states_uc_c.clone()

        n = hidden_states

        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x += n

        if "middle_patch" in transformer_patches:
            patch = transformer_patches["middle_patch"]
            for p in patch:
                x = p(x, extra_options)

        if self.attn2 is not None:
            n = self.norm2(x)
            if self.switch_temporal_ca_to_sa:
                context_attn2 = n
            else:
                context_attn2 = context
            value_attn2 = None
            if "attn2_patch" in transformer_patches:
                patch = transformer_patches["attn2_patch"]
                value_attn2 = context_attn2
                for p in patch:
                    n, context_attn2, value_attn2 = p(n, context_attn2, value_attn2, extra_options)

            attn2_replace_patch = transformer_patches_replace.get("attn2", {})
            block_attn2 = transformer_block
            if block_attn2 not in attn2_replace_patch:
                block_attn2 = block

            if block_attn2 in attn2_replace_patch:
                if value_attn2 is None:
                    value_attn2 = context_attn2
                n = self.attn2.to_q(n)
                context_attn2 = self.attn2.to_k(context_attn2)
                value_attn2 = self.attn2.to_v(value_attn2)
                n = attn2_replace_patch[block_attn2](n, context_attn2, value_attn2, extra_options)
                n = self.attn2.to_out(n)
            else:
                n = self.attn2(n, context=context_attn2, value=value_attn2)

        if "attn2_output_patch" in transformer_patches:
            patch = transformer_patches["attn2_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x += n
        if self.is_res:
            x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        return x


def _get_block_modules(module):
    blocks = list(filter(lambda x: isinstance_str(x[1], 'BasicTransformerBlock'), module.named_modules()))
    return [block for _, block in blocks]


def inject_ref_blocks(diffusion_model: UNetModel):
    input = _get_block_modules(diffusion_model.input_blocks)
    middle = _get_block_modules(diffusion_model.middle_block)
    output = _get_block_modules(diffusion_model.output_blocks)

    for i, block in enumerate(input):
        block.__class__ = RefTransformerBlock
        block.configure('input', i)

    for i, block in enumerate(middle):
        block.__class__ = RefTransformerBlock
        block.configure('middle', i)

    for i, block in enumerate(output):
        block.__class__ = RefTransformerBlock
        block.configure('output', i)
