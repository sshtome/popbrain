"""
PopBrain Neuromorphic Spiking Synaptic Adapter
==============================================
Three-factor synaptic plasticity adapter with Leaky Integrate-and-Fire (LIF) dynamics.
Implements reward-modulated spike-timing-dependent plasticity (STDP) for language models
without backpropagation.
"""

from __future__ import annotations

import os
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file


class PopBrainSpikingAdapter(nn.Module):
    """
    Neuromorphic Spiking LIF Synaptic Adapter.
    
    Attributes:
        vocab_size (int): Vocabulary dimension.
        hidden_dim (int): Number of LIF neurons in population.
        tau (float): Membrane decay time constant.
        v_threshold (float): Action potential firing threshold.
        eta (float): Synaptic learning rate for dopamine modulation.
        trace_decay (float): Half-life decay of eligibility traces.
    """
    def __init__(
        self,
        vocab_size: int = 151665,
        hidden_dim: int = 256,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        eta: float = 0.08,
        trace_decay: float = 0.92,
        device: Optional[str] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.tau = float(tau)
        self.decay = 1.0 - (1.0 / self.tau)
        self.v_threshold = float(v_threshold)
        self.eta = float(eta)
        self.trace_decay = float(trace_decay)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Synaptic weight matrix: hidden_dim x hidden_dim
        self.w_syn = nn.Parameter(torch.zeros(hidden_dim, hidden_dim, device=self.device))
        
        # Population projection bias vector
        self.token_synapses = nn.Parameter(torch.zeros(hidden_dim, device=self.device))
        
        # State buffers
        self.register_buffer("v_membrane", torch.zeros(hidden_dim, device=self.device))
        self.register_buffer("eligibility_trace", torch.zeros(hidden_dim, hidden_dim, device=self.device))
        self.register_buffer("token_trace", torch.zeros(hidden_dim, device=self.device))
        
        self.total_reward_accumulated = 0.0
        self.steps_trained = 0

    def reset_episode(self):
        """Resets membrane potential and eligibility traces for an episode."""
        self.v_membrane.zero_()
        self.eligibility_trace.zero_()
        self.token_trace.zero_()

    def step_tokens(self, token_ids: List[int]):
        """
        Integrates incoming tokens, triggers action potentials, and updates eligibility traces.
        """
        if not token_ids:
            return
            
        with torch.no_grad():
            mapped = torch.tensor([t % self.hidden_dim for t in token_ids[-32:]], device=self.device)
            current = torch.zeros(self.hidden_dim, device=self.device)
            current.scatter_add_(0, mapped, torch.ones_like(mapped, dtype=torch.float32))
            
            # Membrane integration: V[t] = V[t-1] * decay + I_ext
            self.v_membrane.mul_(self.decay).add_(current)
            
            # Spike generation
            spikes = (self.v_membrane >= self.v_threshold).float()
            self.v_membrane.masked_fill_(spikes.bool(), 0.0)
            
            # Update eligibility traces: E[t] = lambda * E[t-1] + (S_t outer S_t)
            self.eligibility_trace.mul_(self.trace_decay).add_(torch.outer(spikes, spikes))
            self.token_trace.mul_(self.trace_decay).add_(spikes)

    def apply_dopamine_reward(self, reward: float):
        """
        Executes three-factor synaptic update:
        Delta W_syn = eta * Dopamine * Eligibility_Trace
        """
        with torch.no_grad():
            r = float(max(-1.0, min(1.0, reward)))
            dW = self.eta * r * self.eligibility_trace
            self.w_syn.add_(dW).clamp_(-3.0, 3.0)
            
            d_token = self.eta * r * self.token_trace
            self.token_synapses.add_(d_token).clamp_(-3.0, 3.0)
            
            self.total_reward_accumulated += r
            self.steps_trained += 1

    def compute_steering_bias(self, candidate_tokens: List[int]) -> Dict[int, float]:
        """
        Calculates token logit steering biases from current synaptic states.
        """
        with torch.no_grad():
            biases = {}
            for tok in candidate_tokens:
                slot = tok % self.hidden_dim
                val = float(self.token_synapses[slot].item())
                if abs(val) > 0.05:
                    biases[tok] = round(val * 1.5, 3)
            return biases

    def save_weights(self, file_path: str):
        """Serializes synaptic weights to disk in PyTorch or SafeTensors format."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        state = {
            "w_syn": self.w_syn.data.cpu(),
            "token_synapses": self.token_synapses.data.cpu()
        }
        if file_path.endswith(".safetensors"):
            save_file(state, file_path)
        else:
            torch.save({
                "state_dict": state,
                "metadata": {
                    "hidden_dim": self.hidden_dim,
                    "vocab_size": self.vocab_size,
                    "steps_trained": self.steps_trained,
                    "total_reward": self.total_reward_accumulated,
                    "synaptic_norm": float(torch.norm(self.w_syn).item())
                }
            }, file_path)

    def load_weights(self, file_path: str):
        """Loads trained synaptic weights from disk."""
        if file_path.endswith(".safetensors"):
            state = load_file(file_path)
            self.w_syn.data.copy_(state["w_syn"].to(self.device))
            self.token_synapses.data.copy_(state["token_synapses"].to(self.device))
        else:
            data = torch.load(file_path, map_location=self.device)
            state = data.get("state_dict", data)
            self.w_syn.data.copy_(state["w_syn"].to(self.device))
            self.token_synapses.data.copy_(state["token_synapses"].to(self.device))
            meta = data.get("metadata", {})
            self.steps_trained = meta.get("steps_trained", 0)
            self.total_reward_accumulated = meta.get("total_reward", 0.0)
