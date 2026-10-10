"""
PopBrain Core Engine: High-Performance Population Spiking Neural Network (SNN)

This module provides the primary simulation engine (PopBrain) designed for large-scale,
sparse biological connectomes (tens of thousands of neurons, millions of synapses).

Architectural Highlights:
-------------------------
1. Population Batching (Block-Diagonal Sparse CSR):
   - Multiple independent agent brains ('slots') are packed into a single unified
     Sparse CSR matrix on the GPU.
   - All agents are stepped simultaneously in parallel without inter-agent crosstalk.
   
2. Custom Fused CUDA Kernels (`snn_cuda`):
   - Inlines Sparse Matrix-Vector multiplication (SpMV), leaky membrane potential decay,
     threshold comparison, refractory counting, spike generation, and continuous STDP.
   - Eliminates Python and PyTorch dispatch overhead across time steps.

3. Synaptic Axonal Transmission Delays:
   - Implements circular ring buffers for realistic axonal transmission delays (e.g. 2 ms)
     with zero tensor reallocation during the simulation loop.

4. Dale's Law & Neurotransmitter Sign Balance:
   - Excitatory connections (ACh, Glu) remain strictly positive.
   - Inhibitory connections (GABA) remain strictly negative.
   - Weights are constrained within biological bounds relative to base genotype weights (W0).

5. Three-Factor Continuous STDP & Dopamine Neuromodulation:
   - Synapses undergo real-time Spike-Timing-Dependent Plasticity based on pre/post traces.
   - Plasticity is modulated by global reward signals (dopamine) per organism.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Set, Dict, List, Tuple

# Ensure local CUDA script tools and allocator configuration
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
from torch.utils.cpp_extension import load

_snn_cuda_module = None


def load_snn_cuda():
    """
    Loads the native CUDA extension for PopBrain.
    
    Resolution Strategy:
    1. Ahead-Of-Time (AOT): Attempts to import the precompiled extension
       `popbrain.snn_cuda` (distributed via binary wheels).
    2. Just-In-Time (JIT): If not precompiled, dynamically compiles `csrc/snn_kernel.cu`
       using PyTorch's `cpp_extension.load` with platform-specific flags.
    """
    global _snn_cuda_module
    if _snn_cuda_module is None:
        try:
            # 1. Try precompiled wheel extension
            import popbrain.snn_cuda as _snn_cuda_module
        except ImportError:
            # 2. Fallback to runtime JIT compilation
            from torch.utils.cpp_extension import load
            base_dir = os.path.dirname(os.path.abspath(__file__))
            extra_cflags = ['-O3', '/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH', '/Zc:preprocessor', '/Zm500'] if os.name == 'nt' else ['-O3']
            extra_cuda = ['-O3', '--use_fast_math']
            if os.name == 'nt':
                extra_cuda += [
                    '-allow-unsupported-compiler',
                    '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',
                    '-Xcompiler', '/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',
                    '-Xcompiler', '/Zc:preprocessor',
                    '-Xcompiler', '/Zm500',
                    '-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING'
                ]
            _snn_cuda_module = load(
                name="snn_cuda",
                sources=[os.path.join(base_dir, "csrc", "snn_kernel.cu")],
                verbose=False,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda
            )
    return _snn_cuda_module


def pick_device(device=None) -> torch.device:
    """Selects target computing device (defaulting to CUDA if available)."""
    if device is not None:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class TensorBrainState:
    """
    Unified GPU state container for the entire population of agents.
    
    Fields:
    -------
    N : int
        Total number of neurons across all slots (Nblk * n_slots).
    Nblk : int
        Number of neurons in a single organism's brain.
    S : int
        Total number of synapses across all slots (Sblk * n_slots).
    Sblk : int
        Number of synapses in a single organism's brain.
    col_idx : torch.Tensor
        CSR column indices (target pre-synaptic neuron for each synapse).
    crow : torch.Tensor
        CSR row pointers defining synapse ranges for each post-synaptic neuron.
    W0 : torch.Tensor
        Baseline genotype weights (unmutated baseline).
    W : torch.Tensor
        Active dynamic synaptic weights (updated by STDP, decay, and clamping).
    plastic : torch.Tensor
        Binary mask (1.0 = plastic synapse in learning zones, 0.0 = static connection).
    fly_of_syn : torch.Tensor
        Mapping indicating which organism slot (0..n_slots-1) owns each synapse.
    V : torch.Tensor
        Membrane potential vector of all neurons [mV].
    a : torch.Tensor
        Spike-frequency adaptation potential (AHP) per neuron [mV].
    refrac_until : torch.Tensor
        Timestamps when absolute refractory period expires for each neuron [ms].
    pre_trace : torch.Tensor
        Exponential eligibility trace for pre-synaptic spikes.
    post_trace : torch.Tensor
        Exponential eligibility trace for post-synaptic spikes.
    reward : torch.Tensor
        Global dopamine/reward signal for each organism in [-1.0, 1.0].
    total_spikes : torch.Tensor
        Cumulative spike counter across the entire simulation.
    spike_ring : list[torch.Tensor]
        Circular ring buffer holding delayed spike vectors for axonal delay simulation.
    vreset_v : torch.Tensor
        Pre-allocated reset voltage vector.
    vth_v : torch.Tensor
        Pre-allocated threshold voltage vector.
    fired_out : torch.Tensor
        Temporary output tensor storing spikes emitted in the current time step.
    """
    N: int
    Nblk: int
    S: int
    Sblk: int
    col_idx: torch.Tensor
    W0: torch.Tensor
    W: torch.Tensor
    plastic: torch.Tensor
    fly_of_syn: torch.Tensor
    crow: torch.Tensor
    V: torch.Tensor
    a: torch.Tensor
    refrac_until: torch.Tensor
    pre_trace: torch.Tensor
    post_trace: torch.Tensor
    reward: torch.Tensor
    total_spikes: torch.Tensor
    spike_ring: list[torch.Tensor]
    vreset_v: torch.Tensor
    vth_v: torch.Tensor
    fired_out: torch.Tensor


class PopBrain:
    """
    High-Performance Population SNN Engine.
    
    Simulates N parallel connectomes in unified Sparse CSR memory format, accelerated
    by custom CUDA kernels with built-in STDP, axonal delays, and Dale's Law enforcement.
    """
    def __init__(self, n_slots: int, minibrain_path: str,
                 nt_sign_dict: Optional[dict] = None, dan_zones: Optional[set] = None,
                 gain: float = 0.4, dt_ms: float = 20.0, tau: float = 100.0,
                 Vrest: float = -65.0, Vth: float = -59.0, Vreset: float = -70.0,
                 refrac: float = 3.0, a_inc: float = 2.5, tau_a: float = 120.0,
                 stdp_eta_pot: float = 0.004, stdp_eta_dep: float = 0.004,
                 tau_pre: float = 20.0, tau_post: float = 20.0,
                 dan_gain: float = 40.0, w_decay: float = 1e-5,
                 reward_decay: float = 0.90):
        """
        Parameters:
        -----------
        n_slots : int
            Number of parallel agent brains (population batch size).
        minibrain_path : str
            Path to connectome file (.json or compressed .npz format).
        nt_sign_dict : dict, optional
            Mapping from neurotransmitter string to sign multiplier (e.g. {'Glu': 1.0, 'GABA': -1.0}).
        dan_zones : set, optional
            Neuropil regions subject to dopamine-modulated STDP plasticity (e.g. {'mbout', 'hub'}).
        gain : float
            Log-synaptic weight scaling factor.
        dt_ms : float
            Simulation time step duration in milliseconds (e.g. 1.0 or 20.0 ms).
        tau : float
            Membrane potential leak time constant [ms].
        Vrest, Vth, Vreset : float
            Resting potential, spike firing threshold, and post-spike reset potential [mV].
        refrac : float
            Absolute refractory period duration [ms].
        a_inc, tau_a : float
            Spike-frequency adaptation increment per spike [mV] and adaptation decay time constant [ms].
        stdp_eta_pot, stdp_eta_dep : float
            STDP learning rates for LTP (potentiation) and LTD (depression).
        tau_pre, tau_post : float
            Time constants for pre- and post-synaptic eligibility traces [ms].
        dan_gain : float
            Dopamine neuromodulation gain coefficient.
        w_decay : float
            Synaptic weight decay rate drifting weights back towards W0 baseline.
        reward_decay : float
            Half-life decay rate of dopamine reward signal over time.
        """
        self.device = pick_device()
        self.n_slots = n_slots
        self.dt = dt_ms
        # Calculate axonal transmission delay slot offset in circular buffer (e.g. 2ms delay)
        self.delay = max(1, int(round(2.0 / dt_ms)))

        import numpy as np
        # 1. Load connectome topology from disk
        if minibrain_path.endswith('.json'):
            with open(minibrain_path, 'r', encoding='utf-8') as f:
                mb = json.load(f)
        elif minibrain_path.endswith('.npz'):
            data = np.load(minibrain_path, allow_pickle=True)
            mb = {
                'neurons': json.loads(str(data['neurons_metadata'])),
                'syn': {
                    'p': data['p'],
                    'q': data['q'],
                    'w': data['w']
                }
            }
        else:
            raise ValueError("minibrain_path must be a .json or .npz file")

        # Set default neurotransmitter sign conventions (Dale's Law)
        if nt_sign_dict is None:
            nt_sign_dict = {'Glu': 1.0, 'GABA': -1.0, 'ACh': 1.0, 'ACH': 1.0, 'GLU': 1.0}
        if dan_zones is None:
            dan_zones = set()

        Nblk = len(mb['neurons'])
        self.zones = [n['zone'] for n in mb['neurons']]
        nt = [n['nt'] if n['nt'] in nt_sign_dict else '(empty)' for n in mb['neurons']]
        self.roles = [n.get('role') for n in mb['neurons']]
        self.role_idx: Dict[str, List[int]] = {}
        for i, r in enumerate(self.roles):
            if r:
                self.role_idx.setdefault(r, []).append(i)
        N = Nblk * n_slots

        # 2. Extract single-brain baseline connectivity
        p = torch.tensor(mb['syn']['p'], dtype=torch.int64)  # Post-synaptic neuron indices (rows)
        q = torch.tensor(mb['syn']['q'], dtype=torch.int64)  # Pre-synaptic neuron indices (columns)
        w = torch.tensor(mb['syn']['w'], dtype=torch.float32) # Raw synapse counts/weights
        nt_sign = torch.tensor([nt_sign_dict.get(nt[i], 0.35) for i in range(Nblk)], dtype=torch.float32)
        Sblk = len(p)
        S = Sblk * n_slots

        # Identify plastic synapses (target post-synaptic neuron belongs to a dopamine-modulated zone)
        plastic_blk = torch.tensor(
            [self.zones[j] in dan_zones for j in mb['syn']['q']],
            dtype=torch.float32
        )

        # 3. Sort synapses in row-major order for CSR construction
        key_blk = p * Nblk + q
        order_blk = torch.argsort(key_blk)
        p_sorted = p[order_blk].to(torch.int64).to(self.device)
        q_sorted = q[order_blk].to(torch.int64).to(self.device)

        base = torch.arange(n_slots, dtype=torch.int64) * Nblk
        self.base = base.to(self.device)
        self.base_cpu = base.tolist()
        
        # Calculate signed baseline weights: nt_sign * log(1 + w) * gain
        W0blk_sorted = (nt_sign[p_sorted.cpu()] * torch.log1p(w[order_blk]) * gain).to(torch.float32).to(self.device)
        plastic_sorted = plastic_blk[order_blk].to(self.device)
        
        # 4. Construct unified block-diagonal tensors for all n_slots parallel agents
        total_syn = n_slots * Sblk
        col_idx = torch.empty(total_syn, dtype=torch.int64, device=self.device)
        W0 = torch.empty(total_syn, dtype=torch.float32, device=self.device)
        plastic = torch.empty(total_syn, dtype=torch.float32, device=self.device)
        fly_of_syn = torch.empty(total_syn, dtype=torch.int64, device=self.device)
        
        for k in range(n_slots):
            start = k * Sblk
            end = (k + 1) * Sblk
            # Offset columns by agent base index to prevent inter-agent crosstalk
            col_idx[start:end] = q_sorted + self.base[k]
            W0[start:end] = W0blk_sorted
            plastic[start:end] = plastic_sorted
            fly_of_syn[start:end] = k
            
        W = W0.clone()

        # Build Compressed Row Pointers (crow)
        p_sorted_cpu = p_sorted.cpu().to(torch.int64)
        crow_blk = torch.zeros(Nblk + 1, dtype=torch.int64)
        crow_blk.index_add_(0, p_sorted_cpu, torch.ones(Sblk, dtype=torch.int64))
        crow_blk = torch.cumsum(crow_blk, 0, dtype=torch.int64)
        crow_g = torch.zeros(N + 1, dtype=torch.int64)
        for k in range(n_slots):
            crow_g[k * Nblk:(k + 1) * Nblk + 1] = crow_blk + k * Sblk
        crow = crow_g.to(self.device)

        # 5. Allocate the unified state
        self.state = TensorBrainState(
            N=N, Nblk=Nblk, S=S, Sblk=Sblk,
            col_idx=col_idx,
            W0=W0, W=W, plastic=plastic, fly_of_syn=fly_of_syn, crow=crow,
            V=torch.full((N,), Vrest, dtype=torch.float32, device=self.device),
            a=torch.zeros(N, dtype=torch.float32, device=self.device),
            refrac_until=torch.zeros(N, dtype=torch.float32, device=self.device),
            pre_trace=torch.zeros(N, dtype=torch.float32, device=self.device),
            post_trace=torch.zeros(N, dtype=torch.float32, device=self.device),
            reward=torch.zeros(n_slots, dtype=torch.float32, device=self.device),
            total_spikes=torch.tensor(0.0, device=self.device),
            spike_ring=[torch.zeros(N, dtype=torch.float32, device=self.device) for _ in range(self.delay + 1)],
            vreset_v=torch.full((N,), Vreset, dtype=torch.float32, device=self.device),
            vth_v=torch.full((N,), Vth, dtype=torch.float32, device=self.device),
            fired_out=torch.zeros(N, dtype=torch.float32, device=self.device)
        )

        self._tmp = torch.empty_like(self.state.W0)
        self.snn_cuda = load_snn_cuda()
        self._rebuild_every_ms = 200.0
        self._since_rebuild = 0.0

        # Precompute exponential decay coefficients
        self.Vrest, self.Vth, self.Vreset = Vrest, Vth, Vreset
        self.refrac, self.a_inc = refrac, a_inc
        self.decay = math.exp(-dt_ms / tau)
        self.a_decay = math.exp(-dt_ms / tau_a)
        self.pre_decay = math.exp(-dt_ms / tau_pre)
        self.post_decay = math.exp(-dt_ms / tau_post)
        self.eta_pot, self.eta_dep = stdp_eta_pot, stdp_eta_dep
        self.dan_gain, self.w_decay = dan_gain, w_decay
        self._reward_step_decay = reward_decay ** (dt_ms / 100.0)

        self.t = 0.0
        self.step_idx = 0

        # 6. Sensory Channel & Motor Pooling Infrastructures
        self._ch_idx: list[torch.Tensor] = []
        self._ch_w: list[float] = []
        self._ch_rate: list[float] = []
        self._ch_key: list[tuple] = []
        self._ch_idx_all = torch.empty(0, dtype=torch.long, device=self.device)
        self._ch_w_all = torch.empty(0, dtype=torch.float32, device=self.device)
        self._ch_pr_all = torch.empty(0, dtype=torch.float32, device=self.device)
        
        # Host pinned memory for zero-copy rate mapping
        self._ch_map_idx = torch.empty(0, dtype=torch.long, device=self.device)
        self._ch_rate_cpu = torch.empty(0, dtype=torch.float32, pin_memory=True)
        self._ch_rate_gpu = torch.empty(0, dtype=torch.float32, device=self.device)
        
        self._channels_dirty = False
        self._rates_dirty = False

        self._pool_masks: dict[tuple, torch.Tensor] = {}
        self._pool_rates: dict[tuple, torch.Tensor] = {}
        self._pool_tau_ms = 500.0
        self._pool_dec = math.exp(-dt_ms / self._pool_tau_ms)
        self._active_list = [False] * n_slots
        self._pool_keys: list[tuple] = []
        self._pool_M = torch.empty(0, self.state.Nblk, dtype=torch.float32, device=self.device)
        self._pool_fly = torch.empty(0, dtype=torch.int64, device=self.device)
        self._pool_rates_t = torch.empty(0, dtype=torch.float32, device=self.device)
        self._pool_arange = torch.empty(0, dtype=torch.int64, device=self.device)
        self._pools_dirty = False

    def _build_csr(self):
        """Constructs a standard PyTorch Sparse CSR tensor wrapping current state."""
        return torch.sparse_csr_tensor(
            self.state.crow, self.state.col_idx, self.state.W,
            size=(self.state.N, self.state.N), device=self.device
        )

    def _rebuild_csr(self):
        """Re-initializes sparse matrix references after structural changes."""
        self.Wcsr = self._build_csr()
        self._since_rebuild = 0.0

    def _rebuild_channels(self):
        """Consolidates registered input channels into unified vectorized GPU tensors."""
        if not self._ch_idx:
            self._ch_idx_all = torch.empty(0, dtype=torch.long, device=self.device)
            self._ch_w_all = torch.empty(0, dtype=torch.float32, device=self.device)
            self._ch_pr_all = torch.empty(0, dtype=torch.float32, device=self.device)
            self._ch_map_idx = torch.empty(0, dtype=torch.long, device=self.device)
            self._ch_rate_cpu = torch.empty(0, dtype=torch.float32, pin_memory=True)
            self._ch_rate_gpu = torch.empty(0, dtype=torch.float32, device=self.device)
            return
            
        self._ch_idx_all = torch.cat(self._ch_idx)
        w_all, map_idx = [], []
        
        for i, (idx, w) in enumerate(zip(self._ch_idx, self._ch_w)):
            n_idx = len(idx)
            w_all.append(torch.full((n_idx,), w, dtype=torch.float32, device=self.device))
            map_idx.append(torch.full((n_idx,), i, dtype=torch.long, device=self.device))
            
        self._ch_w_all = torch.cat(w_all)
        self._ch_map_idx = torch.cat(map_idx)
        
        num_ch = len(self._ch_idx)
        self._ch_rate_cpu = torch.zeros(num_ch, dtype=torch.float32, pin_memory=True)
        self._ch_rate_gpu = torch.zeros(num_ch, dtype=torch.float32, device=self.device)
        
        for i, rate in enumerate(self._ch_rate):
            self._ch_rate_cpu[i] = float(rate)
            
        self._apply_rates()

    # ---------------------------------------------------------------- Slot Management
    def revive(self, fly: int):
        """Activates organism slot, resetting voltages, refractory timers, and adaptation."""
        b = self.base_cpu[fly]
        self._active_list[fly] = True
        self.state.refrac_until[b:b + self.state.Nblk] = 0.0
        self.state.V[b:b + self.state.Nblk] = self.Vrest
        self.state.a[b:b + self.state.Nblk] = 0.0
        self.state.reward[fly] = 0.0

    def kill(self, fly: int):
        """Deactivates organism slot by placing all neurons into infinite refractory clamping."""
        b = self.base_cpu[fly]
        self._active_list[fly] = False
        self.state.refrac_until[b:b + self.state.Nblk] = 1e18
        self.state.reward[fly] = 0.0

    def get_block_weights(self, fly: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns detached copies of active weights (W) and baseline weights (W0) for a slot."""
        a0, a1 = fly * self.state.Sblk, (fly + 1) * self.state.Sblk
        return self.state.W[a0:a1].detach().cpu().clone(), self.state.W0[a0:a1].detach().cpu().clone()

    def set_block_weights(self, fly: int, W_blk: torch.Tensor, W0_blk: torch.Tensor):
        """Sets active and baseline weights for a slot and clamps them according to Dale's Law."""
        a0, a1 = fly * self.state.Sblk, (fly + 1) * self.state.Sblk
        self.state.W[a0:a1] = W_blk.to(self.device)
        self.state.W0[a0:a1] = W0_blk.to(self.device)
        self._clamp_weights()
        self._rebuild_csr()

    def clone_brain(self, source_fly: int, target_fly: int, sigma: float = 0.1, frac: float = 0.2):
        """Clones synaptic weights from source slot to target slot with mutation."""
        a0, a1 = source_fly * self.state.Sblk, (source_fly + 1) * self.state.Sblk
        b0, b1 = target_fly * self.state.Sblk, (target_fly + 1) * self.state.Sblk
        self.state.W[b0:b1] = self.state.W[a0:a1].clone()
        self.state.W0[b0:b1] = self.state.W0[a0:a1].clone()
        self.mutate_weights(target_fly, sigma=sigma, frac=frac)

    def mutate_weights(self, fly: int, sigma: float = 0.2, frac: float = 0.3):
        """Mutates a random fraction of baseline weights (W0) using Gaussian noise."""
        b0, b1 = fly * self.state.Sblk, (fly + 1) * self.state.Sblk
        n_syn = self.state.Sblk
        mask = torch.rand(n_syn, device=self.device) < frac
        noise = torch.exp(torch.randn(n_syn, device=self.device) * sigma)
        self.state.W0[b0:b1] = torch.where(mask, self.state.W0[b0:b1] * noise, self.state.W0[b0:b1])
        self.state.W[b0:b1] = torch.where(mask, self.state.W0[b0:b1], self.state.W[b0:b1])
        self._clamp_weights()

    # ---------------------------------------------------------------- Channels & Pools
    def add_channel(self, name: str, idxs: List[int], rate: float, w: float, fly: int):
        """
        Registers an input sensory channel mapped to a specific set of neurons.
        
        Spikes are generated continuously following a Poisson process matching `rate` (Hz).
        """
        if (name, fly) in self._ch_key:
            self.set_rate_by_index(self._ch_key.index((name, fly)), rate)
            return
        base = self.base_cpu[fly]
        idx = torch.tensor(idxs, dtype=torch.long) + base
        self._ch_key.append((name, fly))
        self._ch_idx.append(idx.to(self.device))
        self._ch_w.append(float(w))
        self._ch_rate.append(float(rate))
        self._channels_dirty = True

    def commit_channels(self):
        """Commits newly registered input channels to GPU memory buffers."""
        if self._channels_dirty:
            self._rebuild_channels()
            self._channels_dirty = False

    def set_rate(self, name: str, rate: float, fly: int):
        """Sets the continuous Poisson firing rate for a sensory channel."""
        key = (name, fly)
        if key in self._ch_key:
            self.set_rate_by_index(self._ch_key.index(key), float(rate))

    def set_rate_by_index(self, i: int, rate: float):
        """Sets the sensory rate by integer channel index."""
        self._ch_rate[i] = float(rate)
        if self._ch_rate_cpu.numel() > i:
            self._ch_rate_cpu[i] = float(rate)
        self._rates_dirty = True

    def _apply_rates(self):
        """Asynchronously streams updated rates from pinned CPU memory to GPU via DMA."""
        if self._ch_rate_cpu.numel() == 0:
            return
        self._ch_rate_gpu.copy_(self._ch_rate_cpu, non_blocking=True)
        # Precompute per-step Poisson spike probabilities: pr = rate * (dt / 1000)
        self._ch_pr_all = self._ch_rate_gpu[self._ch_map_idx] * (self.dt / 1000.0)
        self._rates_dirty = False

    def add_pool(self, name: str, idxs: List[int], fly: int):
        """
        Registers an output motor pool reading integrated firing activity from neuron indices.
        """
        if (name, fly) in self._pool_keys or (name, fly) in self._pool_masks:
            return
        mask = torch.zeros(self.state.Nblk, dtype=torch.float32, device=self.device)
        mask[torch.tensor(idxs, dtype=torch.long, device=self.device)] = 1.0
        self._pool_masks[(name, fly)] = mask
        self._pool_rates[(name, fly)] = torch.tensor(0.0, device=self.device)
        self._pools_dirty = True

    def _rebuild_pools(self):
        """Builds vectorized pooling matrix for simultaneous extraction of motor rates."""
        self._pool_keys = list(self._pool_masks.keys())
        if not self._pool_keys:
            self._pool_M = torch.empty(0, self.state.Nblk, dtype=torch.float32, device=self.device)
            self._pool_fly = torch.empty(0, dtype=torch.int64, device=self.device)
            self._pool_rates_t = torch.empty(0, dtype=torch.float32, device=self.device)
            self._pool_arange = torch.empty(0, dtype=torch.int64, device=self.device)
        else:
            self._pool_M = torch.stack([self._pool_masks[k] for k in self._pool_keys])
            self._pool_fly = torch.tensor([k[1] for k in self._pool_keys], dtype=torch.int64, device=self.device)
            self._pool_rates_t = torch.zeros(len(self._pool_keys), dtype=torch.float32, device=self.device)
            self._pool_arange = torch.arange(len(self._pool_keys), device=self.device)
        self._pools_dirty = False

    def pool_rate(self, name: str, fly: int) -> float:
        """Reads the leaky integrated firing rate (Hz) of a registered output motor pool."""
        key = (name, fly)
        if key not in self._pool_keys:
            return 0.0
        idx = self._pool_keys.index(key)
        return float(self._pool_rates_t[idx].item())

    def get_all_pool_rates(self) -> np.ndarray:
        """Returns all motor pool rates as a numpy array."""
        if not self._pool_keys:
            return np.empty(0, dtype=np.float32)
        return self._pool_rates_t.cpu().numpy()

    def reward_signal(self, amount: float, fly: int):
        """Applies dopamine reward reinforcement signal in [-1.0, 1.0] to an organism."""
        self.state.reward[fly].add_(amount).clamp_(min=-1.0, max=1.0)

    def add_rewards_batch(self, amounts: np.ndarray):
        """Batch-updates rewards for all organisms simultaneously."""
        t = torch.tensor(amounts, dtype=torch.float32, device=self.device)
        self.state.reward.add_(t).clamp_(min=-1.0, max=1.0)

    # ---------------------------------------------------------------- Core Simulation
    def _clamp_weights(self):
        """
        Enforces Dale's Law and structural stability:
        - Excitatory weights remain positive in [0, 2 * W0].
        - Inhibitory weights remain negative in [2 * W0, 0].
        """
        torch.mul(self.state.W0, 2.0, out=self._tmp)
        self._tmp.masked_fill_(self.state.W0 <= 0, 0.0)
        torch.minimum(self.state.W, self._tmp, out=self.state.W)
        
        torch.mul(self.state.W0, 2.0, out=self._tmp)
        self._tmp.masked_fill_(self.state.W0 > 0, 0.0)
        torch.maximum(self.state.W, self._tmp, out=self.state.W)

    def step(self, ms: float = 100.0):
        """
        Executes continuous-time simulation for `ms` milliseconds.
        
        Step Lifecycle:
        ---------------
        1. Advances time and fetches incoming spikes from circular delay ring buffer.
        2. Injects stochastic Poisson sensory inputs.
        3. Invokes fused CUDA kernel (`snn_cuda.forward_step`):
           - Integrates sparse synaptic matrix-vector product.
           - Updates leaky membrane potential V.
           - Emits spikes for neurons exceeding threshold.
           - Applies 3-factor STDP synaptic plasticity.
           - Updates exponential pre/post eligibility traces.
        4. Writes emitted spikes into future delay ring buffer slot.
        5. Updates leaky motor pooling rates.
        6. Decays adaptation potentials (AHP) and dopamine signals.
        7. Periodically applies structural weight decay drifting W towards W0.
        """
        steps = max(1, int(round(ms / self.dt)))
        if self._pools_dirty:
            self._rebuild_pools()
        if self._rates_dirty:
            self._apply_rates()
            
        for _ in range(steps):
            self.step_idx += 1
            self.t = self.step_idx * self.dt
            
            # Read delayed spikes from circular ring buffer slot
            slot = self.step_idx % (self.delay + 1)
            spikes = self.state.spike_ring[slot]

            # Inject Poisson sensory channel spikes
            if self._ch_idx_all.numel():
                mask = torch.rand(self._ch_idx_all.shape[0], device=self.device) < self._ch_pr_all
                self.state.V.index_add_(0, self._ch_idx_all, mask.float() * self._ch_w_all)

            # --- CUSTOM FUSED CUDA KERNEL DISPATCH ---
            self.snn_cuda.forward_step(
                self.state.crow,
                self.state.col_idx,
                self.state.W,
                self.state.plastic,
                self.state.fly_of_syn,
                spikes,
                self.state.V,
                self.state.a,
                self.state.refrac_until,
                self.state.pre_trace,
                self.state.post_trace,
                self.state.reward,
                self.state.fired_out,
                float(self.t),
                float(self.Vrest), float(self.Vth), float(self.Vreset), float(self.refrac),
                float(self.decay), float(self.a_inc), float(self.pre_decay), float(self.post_decay),
                float(self.eta_pot), float(self.eta_dep), float(self.dan_gain)
            )

            fired_f = self.state.fired_out

            # Reset current ring slot and schedule emitted spikes for (slot + delay)
            self.state.spike_ring[slot].zero_()
            self.state.total_spikes.add_(fired_f.sum())
            self.state.spike_ring[(slot + self.delay) % (self.delay + 1)].copy_(fired_f)

            # Integrate motor pooling activity
            if self._pool_M.shape[0]:
                n_mat = self._pool_M @ fired_f.view(self.n_slots, self.state.Nblk).T
                n_vec = n_mat[self._pool_arange, self._pool_fly]
                self._pool_rates_t.mul_(self._pool_dec).add_(n_vec, alpha=1000.0 * (1.0 - self._pool_dec) / self.dt)

            # Exponential decay of adaptation potential and dopamine rewards
            self.state.a.mul_(self.a_decay)
            self.state.reward.mul_(self._reward_step_decay)

            # Periodic structural weight decay
            self._since_rebuild += self.dt
            if self._since_rebuild >= self._rebuild_every_ms:
                if self.w_decay > 0:
                    k = self.w_decay * self._rebuild_every_ms
                    self.state.W.mul_(1.0 - k).add_(self.state.W0, alpha=k)
                    self._clamp_weights()
                self._since_rebuild = 0.0

    def avg_rate(self) -> float:
        """Returns population mean firing rate (Hz) per active organism."""
        n_active = max(1, self._active_list.count(True))
        return float(self.state.total_spikes.item()) / max(self.t, 1e-6) * 1000.0 / (self.state.Nblk * n_active)

    def stats(self) -> dict:
        """Returns diagnostic simulation telemetry summary dictionary."""
        return {
            't': self.t,
            'total_spikes': int(self.state.total_spikes.item()),
            'avg_rate_hz': round(self.avg_rate(), 2),
            'device': str(self.device),
            'w_mean': round(float(self.state.W.mean().item()), 4),
            'active': self._active_list.count(True),
        }
