"""
PopBrain Utilities: Topology Converters and File Handlers

This module provides serialization utilities for converting textual/JSON connectomes
into binary compressed NumPy archives (.npz), reducing disk size and accelerating
startup times for massive networks by up to 20x.
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np


def convert_json_to_npz(json_path: str, npz_path: str):
    """
    Converts a PopBrain JSON connectome into a highly compressed NPZ binary archive.
    
    Structure of the NPZ archive:
    - `p`: int32 array of post-synaptic neuron indices.
    - `q`: int32 array of pre-synaptic neuron indices.
    - `w`: float32 array of synaptic weights.
    - `neurons_metadata`: JSON-encoded string of neuron properties (zones, neurotransmitters, roles).
    
    Parameters:
    -----------
    json_path : str
        Source JSON connectome file path.
    npz_path : str
        Destination binary .npz file path.
    """
    print(f"Loading {json_path} (this might take a while for large files)...")
    with open(json_path, 'r', encoding='utf-8') as f:
        mb = json.load(f)
    
    print("Converting arrays to 32-bit compact binary representations...")
    p = np.array(mb['syn']['p'], dtype=np.int32)
    q = np.array(mb['syn']['q'], dtype=np.int32)
    w = np.array(mb['syn']['w'], dtype=np.float32)
    
    neurons_meta = json.dumps(mb['neurons'])
    
    print(f"Saving compressed archive to {npz_path}...")
    np.savez_compressed(npz_path, p=p, q=q, w=w, neurons_metadata=neurons_meta)
    
    json_sz = os.path.getsize(json_path) / (1024 * 1024)
    npz_sz = os.path.getsize(npz_path) / (1024 * 1024)
    print(f"Conversion complete! Disk size reduced from {json_sz:.2f} MB to {npz_sz:.2f} MB ({json_sz/npz_sz:.1f}x compression).")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python -m popbrain.utils input.json output.npz")
    else:
        convert_json_to_npz(sys.argv[1], sys.argv[2])
