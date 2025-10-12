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


"""
This file is a small demo to show how to merge the pre-trained checkpoints to a whole and load it to device. 
This is what you can do immediately after downloading the pre-traiend checkpoints from HuggingFace.
After this execution a config.json file will be saved. 
"""

from argparse import ArgumentParser
from safetensors import safe_open
from safetensors.torch import save_file
import os
import json
import dataclasses
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config

def merge_checkpoints(safetensors_files, output_dir):
    """
    Merge multiple safetensors checkpoint files and save in the correct format for PI0.
    
    Args:
        safetensors_files (list): List of paths to the sharded .safetensors files
        output_dir (str): Directory where to save the merged model files
    """
    print(f"")
    
    # Dictionary to store all tensors
    merged_tensors = {}
    
    # Load each sharded file and collect tensors
    for file_path in safetensors_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in merged_tensors:
                    raise KeyError(f"Duplicate tensor key '{key}' found in {file_path}.")
                merged_tensors[key] = f.get_tensor(key)
    
    # Create the pretrained_model directory
    pretrained_dir = os.path.join(output_dir, "pretrained_model")
    os.makedirs(pretrained_dir, exist_ok=True)
    
    # Save the merged tensors as model.safetensors
    model_path = os.path.join(pretrained_dir, "model.safetensors")
    save_file(merged_tensors, model_path)
    
    # Save the config.json
    config = PI0Config()
    config_path = os.path.join(pretrained_dir, "config.json")
    
    with open(config_path, 'w') as f:
        json.dump(dataclasses.asdict(config), f, indent=2)
    
    print(f"Merged model saved to {pretrained_dir}")
    print(f"- Model weights: {model_path}")
    print(f"- Config file: {config_path}")
    
    return pretrained_dir, config_path

def load_pi0_model(checkpoint_dir, device=None):
    """
    Load a Pi0 model from a checkpoint directory.
    
    Args:
        checkpoint_dir (str): Path to the directory containing the model checkpoint
        device (str, optional): The device to load the model on. If None, the device from the 
            config file or auto-selection will be used.
    
    Returns:
        PI0Policy: The loaded model
    """
    config = PI0Config.from_pretrained(checkpoint_dir)
    if device is not None:
        config.device = device

    policy = PI0Policy.from_pretrained(checkpoint_dir, config=config)
    return policy

if __name__ == "__main__":
    parser=ArgumentParser(
        description='Description of the script for loading pi-zero pre-trained checkpoint to some device.'
    )
    parser.add_argument('--device', type=str, default=None, help='Device to load your pi-zero model. Can be cpu, cuda, or cuda:x')
    parser.add_argument('--safetensors_files', 
                        type=list, 
                        default=[
                            "physical-intelligence/pi0_base/model-00001-of-00003.safetensors",
                            "physical-intelligence/pi0_base/model-00002-of-00003.safetensors",
                            "physical-intelligence/pi0_base/model-00003-of-00003.safetensors",
                        ], 
                        help='A list of files that stores your model checkpoints, which is downloaded from HuggingFace. \
                            The paths are usually in a list due to space constraint.')
    parser.add_argument('--output_dir', type=str, default='physical-intelligence/pi0_base', 
                        help='the path to the merged full checkpoint, which is stored as a safetensor.')
    args=parser.parse_args()
    
    # Paths to the sharded .safetensors files
    device = args.device
    safetensors_files = args.safetensors_files
    output_dir = args.output_dir
    
    # Output directory for the merged model
    os.makedirs(output_dir, exist_ok=True)
    
    # # Merge the checkpoints and get the pretrained model directory
    pretrained_dir, config_path = merge_checkpoints(safetensors_files, output_dir)
    # Load the model
    try:
        # Set the device to load the model on, for example: "cuda:1" or "cpu"
        model = load_pi0_model(pretrained_dir, device=device)
        print(f"Successfully loaded Pi0 model on {model.config.device}, with the configuration file saved to {config_path}")
    except Exception as e:
        print(f"Error loading model: {e}")