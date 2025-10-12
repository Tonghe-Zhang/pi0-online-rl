# MIT License

# Copyright (c) 2025 Tonghe Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import gc 
import torch 
import sys

def get_memory_usage():
    """Get memory usage for all available GPUs"""
    if not torch.cuda.is_available():
        return {}
    
    memory_info = {}
    for device_id in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3    # GB
        total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3  # GB
        memory_info[f"gpu_{device_id}"] = {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "utilization_percent": (allocated / total) * 100
        }
    return memory_info

def print_memory_usage():
    """Print current memory usage for all GPUs"""
    memory_info = get_memory_usage()
    if memory_info:
        print("Current GPU Memory Usage:")
        for gpu_id, info in memory_info.items():
            print(f"  {gpu_id}: {info['allocated_gb']:.2f}GB allocated, {info['reserved_gb']:.2f}GB reserved, {info['utilization_percent']:.1f}% utilization")
    else:
        print("No CUDA devices available")

def cleanup_cuda_memory():
    """Clean up CUDA memory and cache on all available GPUs"""
    print("Cleaning up residual CUDA memory...")
    gc.collect()
    
    # Handle CUDA cleanup safely in multiprocessing context
    if torch.cuda.is_available():
        try:
            # Get the current device to restore it later
            current_device = torch.cuda.current_device()
            device_count = torch.cuda.device_count()
            
            # Clean up memory on all GPUs
            for device_id in range(device_count):
                torch.cuda.set_device(device_id)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Force garbage collection for this device
                if hasattr(torch.cuda, 'memory_summary'):
                    # Print memory summary for debugging
                    print(f"GPU {device_id} memory before cleanup: {torch.cuda.memory_allocated(device_id) / 1024**3:.2f} GB")
                
                print(f"CUDA memory cleaned up on GPU {device_id}")
            
            # Restore the original device
            torch.cuda.set_device(current_device)
            
            # Additional aggressive cleanup
            gc.collect()
            
            # Force PyTorch to release unused memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print(f"CUDA memory cleaned up on all {device_count} GPUs")
        except RuntimeError as e:
            if "Cannot re-initialize CUDA in forked subprocess" in str(e):
                print("Skipping CUDA cleanup in forked subprocess (this is normal)")
            else:
                print(f"Warning: CUDA cleanup failed: {e}")
        except Exception as e:
            print(f"Warning: CUDA cleanup failed: {e}")
        

def signal_handler(signum, frame):
    """Handle interrupt signals to ensure proper cleanup"""
    print(f"\nReceived signal {signum}. Cleaning up...")
    cleanup_cuda_memory()
    sys.exit(0)