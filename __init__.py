from .nodes.configure_ref_net_node import ConfigureRefNetNode
from .nodes.prepare_ref_latents import PrepareRefLatentsNode
from .nodes.read_sampler_node import ReadSamplerNode
from .nodes.write_sampler_node import WriteSamplerNode
from .nodes.ref_bank_node import CreateRefBankNode
from .nodes.ref_model_pred_node import RefModelSamplingPredNode
from .nodes.custom_ref_map_node import ConfigRefMapAdvNode, CustomRefMapSD1Node


NODE_CLASS_MAPPINGS = {
    "ConfigureRefNet": ConfigureRefNetNode,
    "PrepareRefLatents": PrepareRefLatentsNode,
    "ReadSampler": ReadSamplerNode,
    "WriteSampler": WriteSamplerNode,
    "CreateRefBank": CreateRefBankNode,
    "RefModelSamplingPred": RefModelSamplingPredNode,
    "CustomRefMapSD1": CustomRefMapSD1Node,
    "ConfigRefMapAdv": ConfigRefMapAdvNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigureRefNet": "REF] Configure Model",
    "PrepareRefLatents": "REF] Prep Sampling Latents",
    "ReadSampler": "REF] Read Sampling",
    "WriteSampler": "REF] Write Sampling",
    "CreateRefBank": "REF] Create Bank",
    "RefModelSamplingPred": "REF] Model Sampling Pred",
    "CustomRefMapSD1": "REF] Ref Attn Map SD1",
    "ConfigRefMapAdv": "REF] Ref Attn Map Adv",
}
