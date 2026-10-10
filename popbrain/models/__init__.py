"""
PopBrain Language Models Subpackage
===================================
Specialized SNN and MatMul-Free Large Language Model architectures:
- SpikeGPT216M: Native LIF-driven SpikeGPT model with Event-Driven Linear Gather.
- BitNet2B: Microsoft 1.58-bit Ternary Synaptic LM (2.0B parameters).
- StreamingQwen32B: Sliding-window streaming executor for 32.5B parameter models.
"""

from .spikegpt import SpikeGPT216M, PopBrainNativeLIF, PopBrainSparseLinear
from .bitnet import BitNet2B, unpack_ternary_weights
from .streaming_llm import StreamingQwen32B

__all__ = [
    "SpikeGPT216M",
    "PopBrainNativeLIF",
    "PopBrainSparseLinear",
    "BitNet2B",
    "unpack_ternary_weights",
    "StreamingQwen32B",
]
