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


#!/usr/bin/env python3
"""
Compare two safetensors model files to understand size differences
"""

import os
import sys
from pathlib import Path
import torch
from safetensors import safe_open
import json
from typing import Dict, Any
import numpy as np

def get_file_size(filepath: str) -> float:
    """Get file size in GB"""
    return os.path.getsize(filepath) / (1024**3)

def analyze_safetensors(filepath: str) -> Dict[str, Any]:
    """Analyze a safetensors file and return detailed info"""
    print(f"\nAnalyzing: {filepath}")
    print(f"File size: {get_file_size(filepath):.2f} GB")
    
    tensors = {}
    total_params = 0
    total_size_bytes = 0
    
    # Note: linter may show errors for safetensors API, but it works correctly
    with safe_open(filepath, framework="pt", device="cpu") as f:  # type: ignore
        # Get all tensor names
        tensor_names = f.keys()
        print(f"Number of tensors: {len(tensor_names)}")
        
        for name in tensor_names:
            tensor = f.get_tensor(name)
            num_params = tensor.numel()
            size_bytes = tensor.element_size() * num_params
            
            tensors[name] = {
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'num_params': num_params,
                'size_mb': size_bytes / (1024**2)
            }
            
            total_params += num_params
            total_size_bytes += size_bytes
    
    return {
        'filepath': filepath,
        'file_size_gb': get_file_size(filepath),
        'total_params': total_params,
        'total_size_gb': total_size_bytes / (1024**3),
        'num_tensors': len(tensors),
        'tensors': tensors
    }

def compare_models(model1_path: str, model2_path: str):
    """Compare two safetensors models"""
    print("="*80)
    print("COMPARING SAFETENSORS MODEL FILES")
    print("="*80)
    
    # Analyze both models
    model1_info = analyze_safetensors(model1_path)
    model2_info = analyze_safetensors(model2_path)
    
    print(f"\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    print(f"Model 1 (SFT): {model1_info['file_size_gb']:.2f} GB, {model1_info['total_params']:,} params, {model1_info['num_tensors']} tensors")
    print(f"Model 2 (Pre-trained): {model2_info['file_size_gb']:.2f} GB, {model2_info['total_params']:,} params, {model2_info['num_tensors']} tensors")
    
    # Parameter difference
    param_diff = model2_info['total_params'] - model1_info['total_params']
    print(f"Parameter difference: {param_diff:,} ({param_diff/model2_info['total_params']*100:.1f}% smaller)")
    
    # Size difference
    size_diff = model2_info['file_size_gb'] - model1_info['file_size_gb']
    print(f"Size difference: {size_diff:.2f} GB ({size_diff/model2_info['file_size_gb']*100:.1f}% smaller)")
    
    # Find missing tensors
    model1_tensors = set(model1_info['tensors'].keys())
    model2_tensors = set(model2_info['tensors'].keys())
    
    missing_in_model1 = model2_tensors - model1_tensors
    missing_in_model2 = model1_tensors - model2_tensors
    common_tensors = model1_tensors & model2_tensors
    
    print(f"\n" + "="*80)
    print("TENSOR ANALYSIS")
    print("="*80)
    
    print(f"Common tensors: {len(common_tensors)}")
    print(f"Missing in SFT model: {len(missing_in_model1)}")
    print(f"Missing in Pre-trained model: {len(missing_in_model2)}")
    
    if missing_in_model1:
        print(f"\nTensors missing in SFT model (likely cause of size difference):")
        missing_params = 0
        missing_size = 0
        for name in sorted(missing_in_model1):
            tensor_info = model2_info['tensors'][name]
            print(f"  {name}: {tensor_info['shape']} ({tensor_info['num_params']:,} params, {tensor_info['size_mb']:.1f} MB)")
            missing_params += tensor_info['num_params']
            missing_size += tensor_info['size_mb']
        print(f"  Total missing: {missing_params:,} params, {missing_size:.1f} MB")
    
    if missing_in_model2:
        print(f"\nTensors missing in Pre-trained model:")
        for name in sorted(missing_in_model2):
            tensor_info = model1_info['tensors'][name]
            print(f"  {name}: {tensor_info['shape']} ({tensor_info['num_params']:,} params, {tensor_info['size_mb']:.1f} MB)")
    
    # Check for shape differences in common tensors
    shape_differences = []
    for name in common_tensors:
        shape1 = model1_info['tensors'][name]['shape']
        shape2 = model2_info['tensors'][name]['shape']
        if shape1 != shape2:
            shape_differences.append((name, shape1, shape2))
    
    if shape_differences:
        print(f"\nTensors with different shapes:")
        for name, shape1, shape2 in shape_differences:
            print(f"  {name}: {shape1} vs {shape2}")
    
    # Analyze largest tensors in each model
    print(f"\n" + "="*80)
    print("LARGEST TENSORS")
    print("="*80)
    
    def get_largest_tensors(model_info, top_n=10):
        tensors = [(name, info['num_params'], info['size_mb'], info['dtype']) for name, info in model_info['tensors'].items()]
        return sorted(tensors, key=lambda x: x[1], reverse=True)[:top_n]
    
    print(f"\nTop 10 largest tensors in SFT model:")
    for name, params, size_mb, dtype in get_largest_tensors(model1_info):
        print(f"  {name}: {params:,} params ({size_mb:.1f} MB) [{dtype}]")
    
    print(f"\nTop 10 largest tensors in Pre-trained model:")
    for name, params, size_mb, dtype in get_largest_tensors(model2_info):
        print(f"  {name}: {params:,} params ({size_mb:.1f} MB) [{dtype}]")

    # Check for potential frozen parameters
    print(f"\n" + "="*80)
    print("POTENTIAL ISSUES")
    print("="*80)
    
    if param_diff > 0:
        print(f"⚠️  SFT model has {param_diff:,} fewer parameters than pre-trained model!")
        print("   This suggests that some layers may not have been saved properly.")
        print("   Possible causes:")
        print("   - Frozen parameters not being saved")
        print("   - Partial model saving (only trainable parameters)")
        print("   - Different model architecture")
        print("   - Checkpoint corruption")
    
    if missing_in_model1:
        print(f"⚠️  {len(missing_in_model1)} tensors are missing from the SFT model")
        print("   This could explain the size difference.")
    
    return model1_info, model2_info

def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Usage: python compare_model_checkpoints.py <sft_model_path> <pretrained_model_path>")
        print("Example: python compare_model_checkpoints.py /path/to/sft/model.safetensors /path/to/pretrained/model.safetensors")
        return
    
    sft_model_path = sys.argv[1]
    pretrained_model_path = sys.argv[2]
    
    # Check if files exist
    if not os.path.exists(sft_model_path):
        print(f"Error: SFT model file not found: {sft_model_path}")
        return
    
    if not os.path.exists(pretrained_model_path):
        print(f"Error: Pre-trained model file not found: {pretrained_model_path}")
        return
    
    # Compare models
    compare_models(sft_model_path, pretrained_model_path)

if __name__ == "__main__":
    main() 