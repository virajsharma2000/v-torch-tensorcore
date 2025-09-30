import torch
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("Supports Tensor Cores:", torch.cuda.get_device_capability(0)[0] >= 7)