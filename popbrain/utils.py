import json
import numpy as np

def convert_json_to_npz(json_path: str, npz_path: str):
    """
    Converts a PopBrain JSON connectome into a highly compressed NPZ format.
    This significantly reduces loading times and disk space for networks with millions of synapses.
    """
    print(f"Loading {json_path} (this might take a while for large files)...")
    with open(json_path, 'r', encoding='utf-8') as f:
        mb = json.load(f)
    
    print("Converting arrays...")
    p = np.array(mb['syn']['p'], dtype=np.int32)
    q = np.array(mb['syn']['q'], dtype=np.int32)
    w = np.array(mb['syn']['w'], dtype=np.float32)
    
    neurons_meta = json.dumps(mb['neurons'])
    
    print(f"Saving to {npz_path}...")
    np.savez_compressed(npz_path, p=p, q=q, w=w, neurons_metadata=neurons_meta)
    
    # Calculate sizes
    import os
    json_sz = os.path.getsize(json_path) / (1024*1024)
    npz_sz = os.path.getsize(npz_path) / (1024*1024)
    print(f"Done! Size reduced from {json_sz:.2f} MB to {npz_sz:.2f} MB.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m popbrain.utils input.json output.npz")
    else:
        convert_json_to_npz(sys.argv[1], sys.argv[2])
