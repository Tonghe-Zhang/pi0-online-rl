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
Convert Pi0 training checkpoints from .pt to safetensors format
"""

import torch
import sys
import argparse
from pathlib import Path
from safetensors.torch import save_file
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_checkpoint_to_safetensors(checkpoint_path: str, output_dir: str = None):
    """Convert a .pt checkpoint to safetensors format
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        output_dir: Directory to save safetensors files (default: same as checkpoint)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = checkpoint_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract metadata
    metadata = {
        'step': str(checkpoint.get('step', 0)),
        'loss': str(checkpoint.get('loss', 0.0)),
        'format': 'pi0_sft_checkpoint'
    }
    
    # Save main model
    if 'model_state_dict' in checkpoint:
        model_path = output_dir / f"{checkpoint_path.stem}_model.safetensors"
        logger.info(f"Saving main model to {model_path}")
        save_file(checkpoint['model_state_dict'], model_path, metadata=metadata)
    
    # Save EMA model if available
    if 'ema_model_state_dict' in checkpoint:
        ema_path = output_dir / f"{checkpoint_path.stem}_ema_model.safetensors"
        logger.info(f"Saving EMA model to {ema_path}")
        save_file(checkpoint['ema_model_state_dict'], ema_path, metadata=metadata)
    
    # Save optimizer and scheduler states (these need to stay as .pt since safetensors is for model weights)
    training_state = {
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', 0.0),
        'optimizer_state_dict': checkpoint.get('optimizer_state_dict', {}),
        'scheduler_state_dict': checkpoint.get('scheduler_state_dict', {}),
    }
    
    if training_state['optimizer_state_dict'] or training_state['scheduler_state_dict']:
        training_path = output_dir / f"{checkpoint_path.stem}_training_state.pt"
        logger.info(f"Saving training state to {training_path}")
        torch.save(training_state, training_path)
    
    logger.info("Conversion completed successfully!")
    return {
        'model_path': output_dir / f"{checkpoint_path.stem}_model.safetensors",
        'ema_path': output_dir / f"{checkpoint_path.stem}_ema_model.safetensors" if 'ema_model_state_dict' in checkpoint else None,
        'training_state_path': output_dir / f"{checkpoint_path.stem}_training_state.pt"
    }

def load_model_from_safetensors(model_path: str, ema_path: str = None):
    """Load model weights from safetensors format
    
    Args:
        model_path: Path to main model safetensors file
        ema_path: Path to EMA model safetensors file (optional)
    
    Returns:
        dict with model_state_dict and ema_model_state_dict (if available)
    """
    from safetensors.torch import load_file
    
    result = {}
    
    # Load main model
    logger.info(f"Loading main model from {model_path}")
    result['model_state_dict'] = load_file(model_path)
    
    # Load EMA model if provided
    if ema_path and Path(ema_path).exists():
        logger.info(f"Loading EMA model from {ema_path}")
        result['ema_model_state_dict'] = load_file(ema_path)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Convert Pi0 checkpoints to safetensors format")
    parser.add_argument("checkpoint_path", help="Path to the .pt checkpoint file")
    parser.add_argument("--output_dir", help="Output directory (default: same as checkpoint)")
    parser.add_argument("--batch", nargs="+", help="Convert multiple checkpoints")
    
    args = parser.parse_args()
    
    if args.batch:
        # Convert multiple checkpoints
        for checkpoint_path in args.batch:
            try:
                convert_checkpoint_to_safetensors(checkpoint_path, args.output_dir)
            except Exception as e:
                logger.error(f"Failed to convert {checkpoint_path}: {e}")
    else:
        # Convert single checkpoint
        try:
            paths = convert_checkpoint_to_safetensors(args.checkpoint_path, args.output_dir)
            print(f"Converted checkpoint saved to:")
            print(f"  Model: {paths['model_path']}")
            if paths['ema_path']:
                print(f"  EMA Model: {paths['ema_path']}")
            print(f"  Training State: {paths['training_state_path']}")
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main() 