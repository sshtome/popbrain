"""
PopBrain Dopamine Neuromodulation Submodule
===========================================
Biological three-factor learning, synaptic plasticity adapters, and runtime reward evaluation.
"""

from popbrain.dopamine.adapter import PopBrainSpikingAdapter
from popbrain.dopamine.synaptic_memory import PopBrainSynapticMemory
from popbrain.dopamine.capacity_evaluator import SynapticCapacityBenchmark
from popbrain.dopamine.hierarchical_memory import PopBrainHierarchicalSynapticMemory

__all__ = [
    "PopBrainSpikingAdapter",
    "PopBrainSynapticMemory",
    "PopBrainHierarchicalSynapticMemory",
    "SynapticCapacityBenchmark"
]
