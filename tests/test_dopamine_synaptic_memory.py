"""
Unit Tests for PopBrain Dopamine & Synaptic Memory Modules
==========================================================
Verifies:
1. PopBrainSpikingAdapter (LIF dynamics, 3-factor reward update, checkpointing).
2. PopBrainSynapticMemory (Hetero-associative sequence ingestion, GABAergic lateral inhibition, gating).
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath("."))

from popbrain.dopamine import (
    PopBrainSpikingAdapter,
    PopBrainSynapticMemory,
    PopBrainHierarchicalSynapticMemory
)


class TestPopBrainDopamineModules(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_spiking_adapter_lifecycle(self):
        adapter = PopBrainSpikingAdapter(vocab_size=1000, hidden_dim=64, device=self.device)
        self.assertEqual(adapter.hidden_dim, 64)

        # Step tokens
        adapter.step_tokens([10, 25, 42])
        self.assertGreater(adapter.eligibility_trace.norm().item(), 0.0)

        # Apply reward
        prev_norm = adapter.w_syn.norm().item()
        adapter.apply_dopamine_reward(reward=1.0)
        new_norm = adapter.w_syn.norm().item()
        self.assertGreater(new_norm, prev_norm)

        # Steering bias
        biases = adapter.compute_steering_bias([10, 25])
        self.assertIsInstance(biases, dict)

    def test_synaptic_memory_lateral_inhibition(self):
        dim = 128
        mem = PopBrainSynapticMemory(
            hidden_dim=dim,
            inhibition_threshold=0.85,
            modulation_strength=0.20,
            device=self.device
        )

        # Create dummy sequence
        seq_len = 8
        h_seq = torch.randn(seq_len, dim, device=self.device, dtype=torch.bfloat16)
        e_seq = torch.randn(seq_len, dim, device=self.device, dtype=torch.bfloat16)

        # Ingest sequence (zero backprop)
        mem.ingest_sequence(h_seq, e_seq, reward=1.0)
        self.assertEqual(mem.num_memories, seq_len - 1)

        # Query with exact matching state (h_seq[2])
        delta_h, max_sim = mem.compute_synaptic_modulation(h_seq[2])
        self.assertGreaterEqual(max_sim, 0.99)
        self.assertGreater(delta_h.norm().item(), 0.0)

        # Query with random orthogonal state (should be inhibited to 0.0)
        random_h = torch.randn(dim, device=self.device, dtype=torch.bfloat16)
        delta_rand, rand_sim = mem.compute_synaptic_modulation(random_h)
        self.assertLess(rand_sim, 0.50)
        self.assertEqual(delta_rand.norm().item(), 0.0)

    def test_synaptic_memory_save_load(self):
        dim = 64
        mem1 = PopBrainSynapticMemory(hidden_dim=dim, device=self.device)
        h_seq = torch.randn(4, dim, device=self.device, dtype=torch.bfloat16)
        e_seq = torch.randn(4, dim, device=self.device, dtype=torch.bfloat16)
        mem1.ingest_sequence(h_seq, e_seq, reward=1.0)

        ckpt_path = "tests/temp_test_memory.pt"
        try:
            mem1.save_memory(ckpt_path)
            self.assertTrue(os.path.exists(ckpt_path))

            mem2 = PopBrainSynapticMemory(hidden_dim=dim, device=self.device)
            mem2.load_memory(ckpt_path)
            self.assertEqual(mem2.num_memories, mem1.num_memories)
            self.assertTrue(torch.allclose(mem1.memory_keys, mem2.memory_keys))
        finally:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

    def test_hierarchical_memory_trajectory_and_margin(self):
        dim = 64
        h_mem = PopBrainHierarchicalSynapticMemory(
            hidden_dim=dim,
            window_size=3,
            gamma=0.5,
            inhibition_threshold=0.80,
            margin_threshold=0.03,
            num_clusters=4,
            device=self.device
        )

        # Ingest sequences
        h_seq1 = torch.randn(6, dim, device=self.device, dtype=torch.bfloat16)
        e_seq1 = torch.randn(6, dim, device=self.device, dtype=torch.bfloat16)
        h_mem.ingest_sequence(h_seq1, e_seq1, reward=1.0)
        self.assertEqual(h_mem.num_memories, 5)

        # Query with exact trajectory
        h_mem.reset_runtime_trajectory()
        for t in range(4):
            delta_h, max_sim = h_mem.compute_synaptic_modulation(h_seq1[t])

        # Step 3 matches key at step 3 with full confidence
        self.assertGreater(max_sim, 0.90)

        # Query with random noise should produce 0.0 modulation (inhibited)
        h_mem.reset_runtime_trajectory()
        rand_h = torch.randn(dim, device=self.device, dtype=torch.bfloat16)
        delta_rand, rand_sim = h_mem.compute_synaptic_modulation(rand_h)
        self.assertEqual(delta_rand.norm().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
