"""
PopBrain Dopamine Neuromodulation Submodule
===========================================
Biological three-factor learning, synaptic plasticity adapters, and runtime reward evaluation.
"""

from popbrain.dopamine.adapter import PopBrainSpikingAdapter
from popbrain.dopamine.synaptic_memory import PopBrainSynapticMemory

__all__ = ["PopBrainSpikingAdapter", "PopBrainSynapticMemory"]
