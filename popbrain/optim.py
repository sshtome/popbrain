import torch
import numpy as np

class AdamES:
    \"\"\"
    OpenAI Evolution Strategies (ES) Optimizer with Adam for PopBrain.
    Uses Antithetic Sampling and Rank Normalization for extreme stability.
    \"\"\"
    def __init__(self, brain, lr=0.002, sigma=0.05, beta1=0.9, beta2=0.999):
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
            # We need an even number of clones for antithetic sampling.
            self.n_clones -= 1
        self.half_n = self.n_clones // 2

    def ask(self, champ_fly=0):
        \"\"\"
        Generates noise (epsilons) and injects the perturbed weights into the clones.
        Slot 0 (or champ_fly) is kept as the pure champion.
        Slots 1 to half_n are positive perturbations.
        Slots half_n+1 to 2*half_n are negative perturbations.
        \"\"\"
        Sblk = self.brain.state.Sblk
        self.champ_W0 = self.brain.state.W0[champ_fly * Sblk : (champ_fly + 1) * Sblk].clone()
        
        # Generate half_n noise vectors
        self.epsilons = torch.randn((self.half_n, Sblk), device=self.brain.device)
        
        # Ensure champion is clean in slot 0
        self.brain.set_block_weights(0, self.champ_W0, self.champ_W0)
        
        # Inject antithetic perturbations
        for i in range(self.half_n):
            pos_W0 = self.champ_W0 + self.sigma * self.epsilons[i]
            neg_W0 = self.champ_W0 - self.sigma * self.epsilons[i]
            
            # Write to slots 1..half_n
            self.brain.set_block_weights(1 + i, pos_W0, pos_W0)
            # Write to slots half_n+1..2*half_n
            self.brain.set_block_weights(1 + self.half_n + i, neg_W0, neg_W0)

    def tell(self, clone_rewards):
        \"\"\"
        Computes the gradient from the clone rewards and applies the Adam update to the champion.
        Expects an array of rewards for slots 1 to n_clones.
        \"\"\"
        if len(clone_rewards) != self.n_clones:
            # Extract exactly the clone rewards if a full array was passed
            clone_rewards = clone_rewards[1 : 1 + self.n_clones]
            
        assert len(clone_rewards) == self.n_clones, "Reward array length mismatch."
        
        # Rank Normalization
        ranks = np.zeros(self.n_clones)
        ranks[np.argsort(clone_rewards)] = np.arange(self.n_clones)
        A = ranks / (self.n_clones - 1.0) - 0.5
        A = (A - np.mean(A)) / (np.std(A) + 1e-8)
        
        A_t = torch.tensor(A, dtype=torch.float32, device=self.brain.device)
        
        A_pos = A_t[:self.half_n]
        A_neg = A_t[self.half_n:]
        A_diff = (A_pos - A_neg).view(self.half_n, 1)
        
        # Compute Gradient
        raw_grad = torch.sum(A_diff * self.epsilons, dim=0) / (self.n_clones * self.sigma)
        
        # Adam Update
        self.t += 1
        if self.m is None:
            self.m = torch.zeros_like(raw_grad)
            self.v = torch.zeros_like(raw_grad)
            
        self.m = self.beta1 * self.m + (1 - self.beta1) * raw_grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (raw_grad ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        adam_grad = m_hat / (torch.sqrt(v_hat) + 1e-8)
        
        # Update Champion
        new_W0 = self.champ_W0 + self.lr * adam_grad
        
        # Write back the new champion to all slots (including 0)
        # Note: In a real training loop, you might want to call ask() again later.
        for i in range(self.n_clones + 1):
            self.brain.set_block_weights(i, new_W0, new_W0)
            
        return new_W0
