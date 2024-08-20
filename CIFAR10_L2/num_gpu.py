import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Print the total number of GPUs available
    print("Number of GPUs:", torch.cuda.device_count())
    
    # Print details for each GPU
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(i)} bytes")
else:
    print("CUDA is not available. Check your installation and drivers.")
