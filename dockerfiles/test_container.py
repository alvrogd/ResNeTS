import torch


if not torch.cuda.is_available():
    raise Exception("CUDA is not available")

print(f"Found {torch.cuda.device_count()} CUDA devices")

current_device = torch.cuda.current_device()

if current_device < 0:
    raise Exception("No CUDA device has been assigned to PyTorch")

print(f"CUDA device {torch.cuda.get_device_name(current_device)} [ID {current_device}, "
      f"{torch.cuda.device(current_device)}] has been assigned to PyTorch")
