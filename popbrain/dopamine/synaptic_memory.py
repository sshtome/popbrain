"""
PopBrain Hetero-Associative Synaptic Memory with Lateral Inhibition
===================================================================
Implements biologically plausible sequence learning and zero-backprop knowledge
retrieval for Large Language Models.

Theoretical Foundations:
------------------------
1. Three-Factor Hebbian Plasticity:
   Associates contextual pre-synaptic neural representations with post-synaptic
   target token directions in single-pass forward propagation (zero backprop).
2. GABAergic Feedforward Lateral Inhibition:
   Implements competitive thresholding to suppress cross-talk and prevent
   common-mode attractor collapse (overcoming the classical Hebbian centroid distortion).
3. Zero-Interference Subspace Preservation:
   When input context does not match learned memory keys (cosine similarity < threshold),
   synaptic modulation is strictly zero, preserving pre-trained model knowledge
   against catastrophic forgetting.
"""

from __future__ import annotations

import os
from typing import Optional, List, Dict, Tuple, Any
import torch
import torch.nn as nn


class PopBrainSynapticMemory(nn.Module):
    """
    Hetero-Associative Spiking Synaptic Memory Module for Language Models.
    
    Attributes:
        hidden_dim (int): Dimensionality of model hidden states.
        inhibition_threshold (float): GABAergic lateral inhibition cutoff (default: 0.85).
        modulation_strength (float): Synaptic feedback scale alpha (default: 0.15).
        device (str): Computation device.
    """
    def __init__(
        self,
        hidden_dim: int = 2560,
        inhibition_threshold: float = 0.85,
        modulation_strength: float = 0.15,
        device: Optional[str] = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inhibition_threshold = inhibition_threshold
        self.modulation_strength = modulation_strength
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Memory store tensors:
        # keys: normalized hidden representations [M, hidden_dim]
        # values: normalized target embedding directions [M, hidden_dim]
        self.register_buffer("memory_keys", torch.empty((0, hidden_dim), device=self.device, dtype=torch.bfloat16))
        self.register_buffer("memory_values", torch.empty((0, hidden_dim), device=self.device, dtype=torch.bfloat16))

        self.num_memories = 0
        self.total_episodes_learned = 0

    def clear(self):
        """Clears all stored synaptic associations."""
        self.memory_keys = torch.empty((0, self.hidden_dim), device=self.device, dtype=torch.bfloat16)
        self.memory_values = torch.empty((0, self.hidden_dim), device=self.device, dtype=torch.bfloat16)
        self.num_memories = 0

    def ingest_sequence(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor,
        reward: float = 1.0
    ):
        """
        Learns a sequence transition online in single forward pass without backprop.
        
        Args:
            hidden_states: Tensor [seq_len, hidden_dim] of hidden representations.
            token_embeddings: Tensor [seq_len, hidden_dim] of token embeddings.
            reward: Dopamine neuromodulatory scalar (+1.0 for valid target facts).
        """
        if reward <= 0.0 or hidden_states.shape[0] < 2:
            return

        with torch.no_grad():
            # Keys: normalized pre-synaptic representations for steps 0 .. T-2
            pre_states = hidden_states[:-1]
            keys_norm = pre_states / (torch.norm(pre_states, dim=-1, keepdim=True) + 1e-6)

            # Values: normalized target post-synaptic embeddings for steps 1 .. T-1
            post_embeds = token_embeddings[1:]
            values_norm = post_embeds / (torch.norm(post_embeds, dim=-1, keepdim=True) + 1e-6)

            if self.num_memories == 0:
                self.memory_keys = keys_norm.clone()
                self.memory_values = values_norm.clone()
            else:
                self.memory_keys = torch.cat([self.memory_keys, keys_norm], dim=0)
                self.memory_values = torch.cat([self.memory_values, values_norm], dim=0)

            self.num_memories = self.memory_keys.shape[0]
            self.total_episodes_learned += 1

    def compute_synaptic_modulation(self, h_last: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Computes lateral-inhibited synaptic modulation vector for current hidden state.
        
        Args:
            h_last: Tensor [hidden_dim] of current final hidden state.
            
        Returns:
            delta_h: Modulated shift vector [hidden_dim].
            max_sim: Maximum cosine similarity with stored keys (for gating diagnostics).
        """
        if self.num_memories == 0:
            return torch.zeros_like(h_last), 0.0

        with torch.no_grad():
            h_norm = h_last / (torch.norm(h_last) + 1e-6)
            # Dot products with all stored memory keys
            sims = torch.mv(self.memory_keys, h_norm)
            max_sim = float(torch.max(sims).item())

            # Lateral Inhibition: GABAergic cutoff
            inhibited = torch.relu(sims - self.inhibition_threshold)
            sum_inhib = inhibited.sum()

            if sum_inhib > 0.0:
                weights = inhibited / sum_inhib
                # Directed modulation towards target post-synaptic representation
                direction = torch.mv(self.memory_values.t(), weights)
                delta_h = self.modulation_strength * direction * torch.norm(h_last)
                return delta_h, max_sim

            return torch.zeros_like(h_last), max_sim

    def save_memory(self, file_path: str):
        """Saves synaptic memory keys and values to PyTorch checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        torch.save({
            "memory_keys": self.memory_keys.cpu(),
            "memory_values": self.memory_values.cpu(),
            "num_memories": self.num_memories,
            "hidden_dim": self.hidden_dim,
            "inhibition_threshold": self.inhibition_threshold,
            "modulation_strength": self.modulation_strength,
            "total_episodes_learned": self.total_episodes_learned
        }, file_path)

    def load_memory(self, file_path: str):
        """Loads synaptic memory checkpoint from disk."""
        data = torch.load(file_path, map_location=self.device)
        self.memory_keys = data["memory_keys"].to(self.device)
        self.memory_values = data["memory_values"].to(self.device)
        self.num_memories = data.get("num_memories", self.memory_keys.shape[0])
        self.hidden_dim = data.get("hidden_dim", self.memory_keys.shape[1])
        self.inhibition_threshold = data.get("inhibition_threshold", self.inhibition_threshold)
        self.modulation_strength = data.get("modulation_strength", self.modulation_strength)
        self.total_episodes_learned = data.get("total_episodes_learned", 1)
