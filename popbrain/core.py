"""
Р‘Р°С‚С‡-РґРІРёР¶РѕРє РїРѕРїСѓР»СЏС†РёРё (PopBrain): N РјСѓС… РІ РѕРґРЅРѕР№ Р±Р»РѕС‡РЅРѕ-РґРёР°РіРѕРЅР°Р»СЊРЅРѕР№ CSR.

РћСЃРѕР±РµРЅРЅРѕСЃС‚Рё:
- РђСЂС…РёС‚РµРєС‚СѓСЂРЅР°СЏ РёР·РѕР»СЏС†РёСЏ: С‚РµРЅР·РѕСЂРЅРѕРµ СЃРѕСЃС‚РѕСЏРЅРёРµ РІС‹РЅРµСЃРµРЅРѕ РІ TensorBrainState.
- In-place РѕРїРµСЂР°С†РёРё PyTorch РґР»СЏ РїСЂРµРґРѕС‚РІСЂР°С‰РµРЅРёСЏ С„СЂР°РіРјРµРЅС‚Р°С†РёРё VRAM.
- РќР°РєРѕРїРёС‚РµР»СЊРЅС‹Р№ STDP: СЃР±РѕСЂ СЃРїР°Р№РєРѕРІ Р·Р° РїРµСЂРёРѕРґ 200 РјСЃ Р±РµР· РїРѕС‚РµСЂСЊ.
- РњСѓС‚Р°С†РёСЏ Р±Р°Р·РѕРІС‹С… РІРµСЃРѕРІ (W0) РґР»СЏ РїСЂР°РІРёР»СЊРЅРѕРіРѕ weight decay.
- РќРёРєР°РєРёС… CPU-GPU СЃРёРЅС…СЂРѕРЅРёР·Р°С†РёР№ РІ РіРѕСЂСЏС‡РёС… С†РёРєР»Р°С… (СѓРґР°Р»РµРЅС‹ .item()).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

os.environ['PATH'] = os.environ.get('PATH', '') + os.pathsep + r'C:\Users\ilya\AppData\Local\Python\pythoncore-3.14-64\Scripts'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
from torch.utils.cpp_extension import load

_snn_cuda_module = None

def load_snn_cuda():
    global _snn_cuda_module
    if _snn_cuda_module is None:
        from torch.utils.cpp_extension import load
        base_dir = os.path.dirname(os.path.abspath(__file__))
        _snn_cuda_module = load(
            name="snn_cuda",
            sources=[os.path.join(base_dir, "csrc", "snn_kernel.cu")],
            verbose=False,
            extra_cflags=['-O3', '/Zc:preprocessor', '/Zm500'],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-Xcompiler', '/Zc:preprocessor', '-Xcompiler', '/Zm500', '-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING']
        )
    return _snn_cuda_module


def pick_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class TensorBrainState:
    """РР·РѕР»РёСЂРѕРІР°РЅРЅРѕРµ С…СЂР°РЅРёР»РёС‰Рµ РІСЃРµРіРѕ С‚РµРЅР·РѕСЂРЅРѕРіРѕ СЃРѕСЃС‚РѕСЏРЅРёСЏ РїРѕРїСѓР»СЏС†РёРё"""
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
    def __init__(self, n_slots: int, minibrain_path: str,
                 nt_sign_dict: dict = None, dan_zones: set = None,
                 gain: float = 0.4, dt_ms: float = 20.0, tau: float = 100.0,
                 Vrest: float = -65.0, Vth: float = -59.0, Vreset: float = -70.0,
                 refrac: float = 3.0, a_inc: float = 2.5, tau_a: float = 120.0,
                 stdp_eta_pot: float = 0.004, stdp_eta_dep: float = 0.004,
                 tau_pre: float = 20.0, tau_post: float = 20.0,
                 dan_gain: float = 40.0, w_decay: float = 1e-5,
                 reward_decay: float = 0.90):
        self.device = pick_device()
        self.n_slots = n_slots
        self.dt = dt_ms
        self.delay = max(1, int(round(2.0 / dt_ms)))

        with open(minibrain_path, 'r', encoding='utf-8') as f:
            mb = json.load(f)
        if nt_sign_dict is None: nt_sign_dict = {'Glu': 1.0, 'GABA': -1.0, 'ACh': 1.0}
        if dan_zones is None: dan_zones = set()
        if nt_sign_dict is None: nt_sign_dict = {'Glu': 1.0, 'GABA': -1.0, 'ACh': 1.0}
        if dan_zones is None: dan_zones = set()

        Nblk = len(mb['neurons'])
        self.zones = [n['zone'] for n in mb['neurons']]
        nt = [n['nt'] if n['nt'] in nt_sign_dict else '(empty)' for n in mb['neurons']]
        self.roles = [n.get('role') for n in mb['neurons']]
        self.role_idx = {}
        for i, r in enumerate(self.roles):
            if r:
                self.role_idx.setdefault(r, []).append(i)
        N = Nblk * n_slots

        # ---- Р±Р°Р·РѕРІС‹Рµ РёРЅРґРµРєСЃС‹ РѕРґРЅРѕРіРѕ Р±Р»РѕРєР° ----
        p = torch.tensor(mb['syn']['p'], dtype=torch.int64)
        q = torch.tensor(mb['syn']['q'], dtype=torch.int64)
        w = torch.tensor(mb['syn']['w'], dtype=torch.float32)
        nt_sign = torch.tensor([nt_sign_dict.get(nt[i], 0.35) for i in range(Nblk)], dtype=torch.float32)
        mag = torch.log1p(w) * gain
        Sblk = len(p)
        S = Sblk * n_slots

        # РїР»Р°СЃС‚РёС‡РЅР°СЏ РјР°СЃРєР° Р±Р»РѕРєР° (РїРѕСЃС‚-РЅРµР№СЂРѕРЅ РІ Р·РѕРЅР°С… РїР°РјСЏС‚Рё)
        plastic_blk = torch.tensor(
            [self.zones[j] in dan_zones for j in mb['syn']['q']],
            dtype=torch.float32)

        # ---- РєРѕРЅРєР°С‚РµРЅР°С†РёСЏ Р±Р»РѕРєРѕРІ (РЅР° CPU, Р·Р°С‚РµРј РЅР° GPU) ----
        key_blk = p * Nblk + q
        order_blk = torch.argsort(key_blk)
        p_sorted = p[order_blk].to(torch.int64).to(self.device)
        q_sorted = q[order_blk].to(torch.int64).to(self.device)

        base = torch.arange(n_slots, dtype=torch.int64) * Nblk
        self.base = base.to(self.device)
        self.base_cpu = base.tolist()
        
        W0blk_sorted = (nt_sign[p_sorted.cpu()] * torch.log1p(w[order_blk]) * gain).to(torch.float32).to(self.device)
        plastic_sorted = plastic_blk[order_blk].to(self.device)
        
        # OOM-proof block initialization
        total_syn = n_slots * Sblk
        col_idx = torch.empty(total_syn, dtype=torch.int64, device=self.device)
        W0 = torch.empty(total_syn, dtype=torch.float32, device=self.device)
        plastic = torch.empty(total_syn, dtype=torch.float32, device=self.device)
        fly_of_syn = torch.empty(total_syn, dtype=torch.int64, device=self.device)
        
        for k in range(n_slots):
            start = k * Sblk
            end = (k + 1) * Sblk
            col_idx[start:end] = q_sorted + self.base[k]
            W0[start:end] = W0blk_sorted
            plastic[start:end] = plastic_sorted
            fly_of_syn[start:end] = k
            
        W = W0.clone()

        p_sorted_cpu = p_sorted.cpu().to(torch.int64)
        crow_blk = torch.zeros(Nblk + 1, dtype=torch.int64)
        crow_blk.index_add_(0, p_sorted_cpu, torch.ones(Sblk, dtype=torch.int64))
        crow_blk = torch.cumsum(crow_blk, 0, dtype=torch.int64)
        crow_g = torch.zeros(N + 1, dtype=torch.int64)
        for k in range(n_slots):
            crow_g[k * Nblk:(k + 1) * Nblk + 1] = crow_blk + k * Sblk
        crow = crow_g.to(self.device)

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
        print("Loading snn_cuda...", flush=True)
        self.snn_cuda = load_snn_cuda()
        print("SNN CUDA loaded.", flush=True)
        self._rebuild_every_ms = 200.0
        self._since_rebuild = 0.0

        # ---- РїР°СЂР°РјРµС‚СЂС‹ РґРёРЅР°РјРёРєРё ----
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

        # ---- РєР°РЅР°Р»С‹ (РѕР±СЉРµРґРёРЅС‘РЅРЅС‹Рµ С‚РµРЅР·РѕСЂС‹) Рё РїСѓР»С‹ ----
        self._ch_idx: list[torch.Tensor] = []
        self._ch_w: list[float] = []
        self._ch_rate: list[float] = []
        self._ch_key: list[tuple] = []
        self._ch_idx_all = torch.empty(0, dtype=torch.long, device=self.device)
        self._ch_w_all = torch.empty(0, dtype=torch.float32, device=self.device)
        self._ch_pr_all = torch.empty(0, dtype=torch.float32, device=self.device)
        
        # Pinned memory tensors for fast rate mapping
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

    # ---------------------------------------------------------------- СѓСЃС‚СЂРѕР№СЃС‚РІРѕ
    def _build_csr(self):
        return torch.sparse_csr_tensor(
            self.state.crow, self.state.col_idx, self.state.W,
            size=(self.state.N, self.state.N), device=self.device)

    def _rebuild_csr(self):
        self.Wcsr = self._build_csr()
        self._since_rebuild = 0.0

    def _rebuild_channels(self):
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

    # ---------------------------------------------------------------- СЃР»РѕС‚С‹
    def revive(self, fly: int):
        b = self.base_cpu[fly]
        self._active_list[fly] = True
        self.state.refrac_until[b:b + self.state.Nblk] = 0.0
        self.state.V[b:b + self.state.Nblk] = self.Vrest
        self.state.a[b:b + self.state.Nblk] = 0.0
        self.state.reward[fly] = 0.0

    def kill(self, fly: int):
        b = self.base_cpu[fly]
        self._active_list[fly] = False
        self.state.refrac_until[b:b + self.state.Nblk] = 1e18
        self.state.reward[fly] = 0.0

    def get_block_weights(self, fly: int):
        a0, a1 = fly * self.state.Sblk, (fly + 1) * self.state.Sblk
        return self.state.W[a0:a1].detach().cpu().clone(), self.state.W0[a0:a1].detach().cpu().clone()

    def set_block_weights(self, fly: int, W_blk, W0_blk):
        a0, a1 = fly * self.state.Sblk, (fly + 1) * self.state.Sblk
        self.state.W[a0:a1] = W_blk.to(self.device)
        self.state.W0[a0:a1] = W0_blk.to(self.device)
        self._clamp_weights()
        self._rebuild_csr()

    def clone_brain(self, source_fly: int, target_fly: int, sigma: float = 0.1, frac: float = 0.2):
        a0, a1 = source_fly * self.state.Sblk, (source_fly + 1) * self.state.Sblk
        b0, b1 = target_fly * self.state.Sblk, (target_fly + 1) * self.state.Sblk
        self.state.W[b0:b1] = self.state.W[a0:a1].clone()
        self.state.W0[b0:b1] = self.state.W0[a0:a1].clone()
        self.mutate_weights(target_fly, sigma=sigma, frac=frac)

    def mutate_weights(self, fly: int, sigma: float = 0.2, frac: float = 0.3):
        b0, b1 = fly * self.state.Sblk, (fly + 1) * self.state.Sblk
        mask = torch.rand(self.state.Sblk, device=self.device) < frac
        noise = 1.0 + torch.randn(self.state.Sblk, dtype=torch.float32, device=self.device) * sigma
        
        self.state.W0[b0:b1] = torch.where(mask, self.state.W0[b0:b1] * noise, self.state.W0[b0:b1])
        self.state.W[b0:b1] = torch.where(mask, self.state.W0[b0:b1], self.state.W[b0:b1])
        self._clamp_weights()

    # ---------------------------------------------------------------- РєР°РЅР°Р»С‹/РїСѓР»С‹
    def add_channel(self, name: str, idxs, rate: float, w: float, fly: int):
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
        if self._channels_dirty:
            self._rebuild_channels()
            self._channels_dirty = False

    def set_rate(self, name: str, rate: float, fly: int):
        key = (name, fly)
        if key in self._ch_key:
            self.set_rate_by_index(self._ch_key.index(key), float(rate))

    def set_rate_by_index(self, i: int, rate: float):
        self._ch_rate[i] = float(rate)
        if self._ch_rate_cpu.numel() > i:
            self._ch_rate_cpu[i] = float(rate)
        self._rates_dirty = True

    def _apply_rates(self):
        if self._ch_rate_cpu.numel() == 0:
            return
        
        # РђСЃРёРЅС…СЂРѕРЅРЅРѕРµ РєРѕРїРёСЂРѕРІР°РЅРёРµ РёР· Pinned Memory РІ VRAM
        self._ch_rate_gpu.copy_(self._ch_rate_cpu, non_blocking=True)
        # Р’РµРєС‚РѕСЂРёР·РѕРІР°РЅРЅС‹Р№ РјР°РїРїРёРЅРі
        self._ch_pr_all = self._ch_rate_gpu[self._ch_map_idx] * (self.dt / 1000.0)
        self._rates_dirty = False

    def add_pool(self, name: str, idxs, fly: int):
        if (name, fly) in self._pool_keys or (name, fly) in self._pool_masks:
            return
        mask = torch.zeros(self.state.Nblk, dtype=torch.float32, device=self.device)
        mask[torch.tensor(idxs, dtype=torch.long, device=self.device)] = 1.0
        self._pool_masks[(name, fly)] = mask
        self._pool_rates[(name, fly)] = torch.tensor(0.0, device=self.device)
        self._pools_dirty = True

    def _rebuild_pools(self):
        self._pool_keys = list(self._pool_masks.keys())
        if not self._pool_keys:
            self._pool_M = torch.empty(0, self.state.Nblk, dtype=torch.float32, device=self.device)
            self._pool_fly = torch.empty(0, dtype=torch.int64, device=self.device)
            self._pool_rates_t = torch.empty(0, dtype=torch.float32, device=self.device)
            self._pool_arange = torch.empty(0, dtype=torch.int64, device=self.device)
        else:
            self._pool_M = torch.stack([self._pool_masks[k] for k in self._pool_keys])
            self._pool_fly = torch.tensor([k[1] for k in self._pool_keys],
                                          dtype=torch.int64, device=self.device)
            self._pool_rates_t = torch.stack([self._pool_rates[k] for k in self._pool_keys])
            self._pool_arange = torch.arange(len(self._pool_keys), dtype=torch.int64,
                                             device=self.device)
        self._pools_dirty = False

    def pool_rate(self, name: str, fly: int) -> float:
        if (name, fly) not in self._pool_keys:
            return 0.0
        p = self._pool_keys.index((name, fly))
        return float(self._pool_rates_t[p].item())

    def pool_rates_numpy(self):
        return self._pool_rates_t.cpu().numpy()

    def reward_signal(self, amount: float, fly: int):
        # In-place, Р±РµР· CPU-GPU СЃРёРЅС…СЂРѕРЅРёР·Р°С†РёРё (.item() СѓРґР°Р»РµРЅ)
        self.state.reward[fly].add_(amount).clamp_(min=-1.0, max=1.0)

    def add_rewards_batch(self, amounts: np.ndarray):
        t = torch.tensor(amounts, dtype=torch.float32, device=self.device)
        self.state.reward.add_(t).clamp_(min=-1.0, max=1.0)

    # ------------------------------------------------------------------ СЏРґСЂРѕ
    def _clamp_weights(self):
        torch.mul(self.state.W0, 2.0, out=self._tmp)
        self._tmp.masked_fill_(self.state.W0 <= 0, 0.0)
        torch.minimum(self.state.W, self._tmp, out=self.state.W)
        
        torch.mul(self.state.W0, 2.0, out=self._tmp)
        self._tmp.masked_fill_(self.state.W0 > 0, 0.0)
        torch.maximum(self.state.W, self._tmp, out=self.state.W)

    def step(self, ms: float = 100.0):
        steps = max(1, int(round(ms / self.dt)))
        if self._pools_dirty:
            self._rebuild_pools()
        if self._rates_dirty:
            self._apply_rates()
            
        for _ in range(steps):
            self.step_idx += 1
            self.t = self.step_idx * self.dt
            slot = self.step_idx % (self.delay + 1)

            spikes = self.state.spike_ring[slot]

            if self._ch_idx_all.numel():
                mask = torch.rand(self._ch_idx_all.shape[0], device=self.device) < self._ch_pr_all
                self.state.V.index_add_(0, self._ch_idx_all, mask.float() * self._ch_w_all)

            # --- CUSTOM CUDA FUSED KERNEL ---
            # Р—Р°РјРµРЅСЏРµС‚ 10+ РІС‹Р·РѕРІРѕРІ PyTorch: decay, sparse matvec, РїРѕСЂРѕРіРё, STDP, РѕР±РЅРѕРІР»РµРЅРёРµ С‚СЂРµР№СЃРѕРІ.
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

            self.state.spike_ring[slot].zero_()
            self.state.total_spikes.add_(fired_f.sum())
            self.state.spike_ring[(slot + self.delay) % (self.delay + 1)].copy_(fired_f)

            if self._pool_M.shape[0]:
                n_mat = self._pool_M @ fired_f.view(self.n_slots, self.state.Nblk).T
                n_vec = n_mat[self._pool_arange, self._pool_fly]
                self._pool_rates_t.mul_(self._pool_dec).add_(n_vec, alpha=1000.0 * (1.0 - self._pool_dec) / self.dt)

            self.state.a.mul_(self.a_decay)
            self.state.reward.mul_(self._reward_step_decay)

            self._since_rebuild += self.dt
            if self._since_rebuild >= self._rebuild_every_ms:
                if self.w_decay > 0:
                    k = self.w_decay * self._rebuild_every_ms
                    self.state.W.mul_(1.0 - k).add_(self.state.W0, alpha=k)
                    self._clamp_weights()
                self._since_rebuild = 0.0

    def avg_rate(self) -> float:
        n_active = max(1, self._active_list.count(True))
        return float(self.state.total_spikes.item()) / max(self.t, 1e-6) * 1000.0 / (self.state.Nblk * n_active)

    def stats(self) -> dict:
        return {
            't': self.t,
            'total_spikes': int(self.state.total_spikes.item()),
            'avg_rate_hz': round(self.avg_rate(), 2),
            'device': str(self.device),
            'w_mean': round(float(self.state.W.mean().item()), 4),
            'active': self._active_list.count(True),
        }






