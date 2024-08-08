

import torch

# Check PyTorch version
print("PyTorch version:", torch.__version__)

# Check CUDA version supported by PyTorch
print("CUDA version supported by PyTorch:", torch.version.cuda)

# Check if CUDA is available
print("Is CUDA available?", torch.cuda.is_available())

# Check the current CUDA device
if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("CUDA capability:", torch.cuda.get_device_capability(torch.cuda.current_device()))

# Check the number of available CUDA devices
print("Number of CUDA devices:", torch.cuda.device_count())

import torch
from torch_geometric.data import Data
