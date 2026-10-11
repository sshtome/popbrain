"""
PopBrain: High-Performance Population Spiking Neural Network Engine for PyTorch.

Public API:
-----------
Core Simulation Engine:
- `PopBrain`: High-performance SNN population simulation engine.
- `TensorBrainState`: GPU state container for all agent connectomes.
- `AdamES`: OpenAI Evolution Strategies optimizer accelerated by Adam.
- `convert_json_to_npz`: Utility for converting JSON connectomes to binary NPZ format.

Streaming & LLM Architecture:
- `SlidingWindowStreamer`: Double-buffered CUDA pipeline for multi-billion parameter LLMs.
- `SpikeActivationGate`: SNN event-driven activation filter (80%+ sparsity inducer).
- `models`: Pretrained SNN / MatMul-free model zoo (SpikeGPT, BitNet, StreamingQwen32B).
"""

from .core import PopBrain, TensorBrainState
from .optim import AdamES
from .utils import convert_json_to_npz
from .streamer import SlidingWindowStreamer, SpikeActivationGate, TieredLayerStreamer
from . import models
from .dopamine import (
    PopBrainSpikingAdapter,
    PopBrainSynapticMemory,
    PopBrainHierarchicalSynapticMemory,
    PopBrainScale10kMemory,
    SynapticCapacityBenchmark,
)

__version__ = "1.3.0"
__all__ = [
    "PopBrain",
    "TensorBrainState",
    "AdamES",
    "convert_json_to_npz",
    "SlidingWindowStreamer",
    "TieredLayerStreamer",
    "SpikeActivationGate",
    "PopBrainSpikingAdapter",
    "PopBrainSynapticMemory",
    "PopBrainHierarchicalSynapticMemory",
    "PopBrainScale10kMemory",
    "SynapticCapacityBenchmark",
    "models",
]
