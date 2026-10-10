"""
PopBrain: High-Performance Population Spiking Neural Network Engine for PyTorch.

Public API:
-----------
- `PopBrain`: Core SNN population simulation engine.
- `TensorBrainState`: GPU state container for all agent brains.
- `AdamES`: OpenAI Evolution Strategies optimizer accelerated by Adam.
- `convert_json_to_npz`: Utility for converting JSON connectomes to binary NPZ format.
"""

from .core import PopBrain, TensorBrainState
from .optim import AdamES
from .utils import convert_json_to_npz

__version__ = "1.0.0"
__all__ = ["PopBrain", "TensorBrainState", "AdamES", "convert_json_to_npz"]
