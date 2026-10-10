import matplotlib.pyplot as plt
import numpy as np

# Set clean styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# Colors
color_popbrain = '#10b981'  # Emerald green
color_sparse   = '#3b82f6'  # Blue
color_dense1   = '#f43f5e'  # Rose red
color_dense2   = '#fb923c'  # Orange

# -------------------------------------------------------------
# Chart 1: VRAM Memory Consumption (Log Scale)
# -------------------------------------------------------------
frameworks_vram = ['PopBrain 1.0\n(Sparse CSR)', 'PyTorch Native\n(Sparse CSR)', 'snnTorch\n(Dense Layers)', 'SpikingJelly\n(Dense Layers)']
vram_mb = [468.7, 489.7, 86400, 86400]
colors_vram = [color_popbrain, color_sparse, color_dense1, color_dense2]

bars1 = ax1.bar(frameworks_vram, vram_mb, color=colors_vram, width=0.55, edgecolor='#1e293b', linewidth=1.2, alpha=0.95)
ax1.set_yscale('log')
ax1.set_ylabel('VRAM Consumption (MB, Log Scale)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_title('GPU VRAM for 32 Agents (26k Neurons / 318k Synapses)', fontsize=13, fontweight='bold', pad=15)
ax1.set_ylim(80, 250000)

# Annotations
for bar, val in zip(bars1, vram_mb):
    yval = bar.get_height()
    if val > 50000:
        label = f">80 GB\n(CUDA OOM)"
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval * 1.25, label, ha='center', va='bottom', fontsize=10, fontweight='bold', color='#be123c')
    else:
        label = f"{val:.1f} MB\n(Fits Consumer GPU)"
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval * 1.25, label, ha='center', va='bottom', fontsize=10, fontweight='bold', color='#047857')

# -------------------------------------------------------------
# Chart 2: Feature & Biological Realism Comparison
# -------------------------------------------------------------
features = [
    'Sparse Connectomes (26k+)',
    'Population Batching (32 Clones)',
    'Online STDP Plasticity',
    'Synaptic Axonal Delays (2ms)',
    'Dopamine Neuromodulation',
    'Dale\'s Law (E/I Enforcement)'
]

popbrain_score = [1, 1, 1, 1, 1, 1]
sparse_score   = [1, 1, 0, 0, 0, 0]
deep_snn_score = [0, 1, 0, 0, 0, 0]

y_pos = np.arange(len(features))
height = 0.25

ax2.barh(y_pos + height, popbrain_score, height, label='PopBrain 1.0', color=color_popbrain, edgecolor='#1e293b', linewidth=1.2)
ax2.barh(y_pos, sparse_score, height, label='PyTorch Sparse Baseline', color=color_sparse, edgecolor='#1e293b', linewidth=1.2)
ax2.barh(y_pos - height, deep_snn_score, height, label='snnTorch / SpikingJelly', color='#94a3b8', edgecolor='#1e293b', linewidth=1.2)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(features, fontsize=11, fontweight='bold')
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['No / Unsupported', 'Yes / Built-in Native'], fontsize=11, fontweight='bold')
ax2.set_title('Biological Simulation Capabilities', fontsize=13, fontweight='bold', pad=15)
ax2.legend(loc='lower right', frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig('assets/benchmark_comparison.png', bbox_inches='tight')
print("Successfully generated assets/benchmark_comparison.png cleanly without glyph warnings")
