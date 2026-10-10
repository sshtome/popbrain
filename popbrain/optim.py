"""
PopBrain Evolutionary Optimizer: Adam-Accelerated Evolution Strategies (AdamES)

This module provides a black-box neuroevolution optimizer based on OpenAI Evolution Strategies (ES)
enhanced with:
1. Antithetic Sampling: Pairs positive (+eps) and negative (-eps) weight perturbations
   for variance reduction.
2. Rank Normalization: Fitness scores are transformed into centered ranks, preventing
   outlier scores from destabilizing optimization.
3. Adam Gradient Updates: Adaptive moment estimation (m, v) with bias correction for
   rapid, smooth convergence.
"""

from __future__ import annotations

import torch
import numpy as np


class AdamES:
    """
    OpenAI Evolution Strategies (ES) Optimizer accelerated by Adam and Antithetic Sampling.
    
    Workflow per Generation:
    -----------------------
    1. `optimizer.ask(champ_fly=0)`:
       Keeps the current Champion in slot 0, then generates `half_n` Gaussian noise vectors.
       Assigns positive perturbations to slots `1..half_n` and symmetric negative perturbations
       to slots `half_n+1..2*half_n`.
    2. User runs parallel simulation loop on `brain.step()`.
    3. User calculates scalar fitness/rewards for each agent.
    4. `optimizer.tell(clone_rewards)`:
       Rank-normalizes the rewards, computes the numerical pseudo-gradient, applies Adam
       momentum and second-moment updates, and writes the improved weights to the Champion.
    """
    def __init__(self, brain, lr: float = 0.002, sigma: float = 0.05, beta1: float = 0.9, beta2: float = 0.999):
        """
        Parameters:
        -----------
        brain : PopBrain
            The PopBrain simulation engine containing population slots.
        lr : float
            Learning rate for the Adam optimizer.
        sigma : float
            Standard deviation of the Gaussian mutation noise.
        beta1 : float
            Adam first-moment exponential decay factor (momentum).
        beta2 : float
            Adam second-moment exponential decay factor (uncentered variance).
        """
        self.brain = brain
        self.lr = lr
        self.sigma = sigma
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.t = 0
        self.epsilons = None
        self.champ_W0 = None
        
        # Determine how many clones we can evaluate
        self.N = brain.state.N // brain.state.Nblk
        if self.N < 3:
            raise ValueError("PopBrain must have at least 3 slots (1 champ, 2 clones) for Antithetic ES.")
        
        self.n_clones = self.N - 1
        if self.n_clones % 2 != 0:
            # We need an even number of clones for symmetric antithetic pairs
            self.n_clones -= 1
        self.half_n = self.n_clones // 2

    def ask(self, champ_fly: int = 0):
        """
        Generates antithetic perturbations and writes them into the population slots.
        
        Slot Layout:
        - Slot 0: Champion (unperturbed baseline).
        - Slots 1 .. half_n: Positive perturbations (W0 + sigma * eps).
        - Slots half_n+1 .. 2*half_n: Negative perturbations (W0 - sigma * eps).
        """
        Sblk = self.brain.state.Sblk
        self.champ_W0 = self.brain.state.W0[champ_fly * Sblk : (champ_fly + 1) * Sblk].clone()
        
        # Sample half_n independent standard Gaussian noise vectors on GPU
        self.epsilons = torch.randn((self.half_n, Sblk), device=self.brain.device)
        
        # Ensure champion is clean in slot 0
        self.brain.set_block_weights(0, self.champ_W0, self.champ_W0)
        
        # Inject symmetric antithetic perturbations across parallel clones
        for i in range(self.half_n):
            pos_W0 = self.champ_W0 + self.sigma * self.epsilons[i]
            neg_W0 = self.champ_W0 - self.sigma * self.epsilons[i]
            
            # Write positive perturbation
            self.brain.set_block_weights(1 + i, pos_W0, pos_W0)
            # Write symmetric negative perturbation
            self.brain.set_block_weights(1 + self.half_n + i, neg_W0, neg_W0)

    def tell(self, clone_rewards):
        """
        Processes evaluation rewards, computes pseudo-gradient, and updates the Champion.
        
        Parameters:
        -----------
        clone_rewards : list or np.ndarray
            Array of scalar rewards evaluated for each clone slot.
        """
        if len(clone_rewards) != self.n_clones:
            # Extract exactly the clone rewards if a full array including champion was passed
            clone_rewards = clone_rewards[1 : 1 + self.n_clones]
            
        assert len(clone_rewards) == self.n_clones, "Reward array length mismatch."
        
        # 1. Rank-based fitness shaping (Salimans et al., OpenAI ES)
        ranks = np.zeros(self.n_clones)
        ranks[np.argsort(clone_rewards)] = np.arange(self.n_clones)
        A = ranks / (self.n_clones - 1.0) - 0.5
        A = (A - np.mean(A)) / (np.std(A) + 1e-8)
        
        A_t = torch.tensor(A, dtype=torch.float32, device=self.brain.device)
        
        # Separate positive and negative antithetic clone utilities
        A_pos = A_t[:self.half_n]
        A_neg = A_t[self.half_n:]
        A_diff = (A_pos - A_neg).view(self.half_n, 1)
        
        # 2. Monte Carlo gradient estimate: grad = sum( (A_pos - A_neg) * eps ) / (n * sigma)
        raw_grad = torch.sum(A_diff * self.epsilons, dim=0) / (self.n_clones * self.sigma)
        
        # 3. Adam Optimizer update with bias correction
        self.t += 1
        if self.m is None:
            self.m = torch.zeros_like(raw_grad)
            self.v = torch.zeros_like(raw_grad)
            
        self.m = self.beta1 * self.m + (1 - self.beta1) * raw_grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (raw_grad ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        adam_grad = m_hat / (torch.sqrt(v_hat) + 1e-8)
        
        # 4. Update the Champion genotype weights
        new_W0 = self.champ_W0 + self.lr * adam_grad
        self.brain.set_block_weights(0, new_W0, new_W0)
        
        return float(torch.norm(adam_grad).item())
