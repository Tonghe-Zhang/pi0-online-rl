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
Setup script for pi0 evaluation on ManiSkill3.
This script helps users install dependencies and download model weights.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description=""):
    """Run a shell command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*50}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    else:
        print(f"Success: {result.stdout}")
        return True


def check_python_packages():
    """Check if required Python packages are installed"""
    required_packages = [
        "torch",
        "transformers",
        "gymnasium", 
        "mani_skill",
        "tyro",
        "numpy",
        "pillow",
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing_packages.append(package)
    
    return missing_packages


def install_dependencies():
    """Install required dependencies"""
    print("Installing required dependencies...")
    
    # Install basic packages
    basic_packages = [
        "torch>=2.0.0",
        "transformers>=4.40.0", 
        "gymnasium",
        "tyro",
        "numpy",
        "pillow",
        "safetensors",
        "huggingface_hub",
    ]
    
    for package in basic_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"Failed to install {package}")
            return False
    
    # Install ManiSkill3
    if not run_command("pip install mani_skill", "Installing ManiSkill3"):
        print("Failed to install ManiSkill3")
        return False
    
    return True


def download_pi0_weights():
    """Download pi0 model weights"""
    model_dir = Path("physical-intelligence/pi0_base")
    
    if model_dir.exists():
        print(f"✓ Pi0 model weights already exist at {model_dir}")
        return True
    
    print("Downloading pi0 model weights...")
    
    # Check if huggingface-cli is available
    result = subprocess.run("which huggingface-cli", shell=True, capture_output=True)
    if result.returncode != 0:
        print("Installing huggingface_hub[cli]...")
        if not run_command("pip install 'huggingface_hub[cli]'", "Installing huggingface_hub[cli]"):
            return False
    
    # Download the model
    cmd = "huggingface-cli download --resume-download yinchenghust/openpi_base --local-dir ./physical-intelligence/pi0_base/"
    if not run_command(cmd, "Downloading pi0 model weights"):
        print("Failed to download pi0 model weights")
        print("You may need to login to HuggingFace first: huggingface-cli login")
        return False
    
    return True


def download_language_tokenizer():
    """Download PaliGemma language tokenizer"""
    print("Downloading PaliGemma language tokenizer...")
    
    cmd = "huggingface-cli download --resume-download google/paligemma-3b-pt-224 --local-dir ./google/paligemma-3b-pt-224"
    if not run_command(cmd, "Downloading PaliGemma tokenizer"):
        print("Failed to download PaliGemma tokenizer")
        print("You may need to login to HuggingFace and accept the license agreement")
        print("Visit: https://huggingface.co/google/paligemma-3b-pt-224")
        return False
    
    return True


def setup_maniskill_assets():
    """Setup ManiSkill asset directory"""
    print("Setting up ManiSkill assets...")
    
    # Create asset directory
    asset_dir = Path.home() / ".maniskill"
    asset_dir.mkdir(exist_ok=True)
    
    # Set environment variable
    env_var = f"export MS_ASSET_DIR={asset_dir}"
    bashrc_path = Path.home() / ".bashrc"
    
    # Check if already set
    if bashrc_path.exists():
        with open(bashrc_path, 'r') as f:
            content = f.read()
            if "MS_ASSET_DIR" in content:
                print("✓ MS_ASSET_DIR already set in .bashrc")
                return True
    
    # Add to .bashrc
    with open(bashrc_path, 'a') as f:
        f.write(f"\n# ManiSkill asset directory\n{env_var}\n")
    
    print(f"✓ Added MS_ASSET_DIR to .bashrc: {asset_dir}")
    print("Please run 'source ~/.bashrc' or restart your terminal")
    
    return True


def main():
    """Main setup function"""
    print("Pi0 ManiSkill3 Evaluation Setup")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check existing packages
    print("\nChecking existing packages...")
    missing_packages = check_python_packages()
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        response = input("Do you want to install missing dependencies? (y/n): ")
        if response.lower() == 'y':
            if not install_dependencies():
                print("Failed to install dependencies")
                sys.exit(1)
        else:
            print("Skipping dependency installation")
    else:
        print("✓ All required packages are installed")
    
    # Download model weights
    print("\nDownloading model weights...")
    if not download_pi0_weights():
        print("Failed to download pi0 model weights")
        sys.exit(1)
    
    # Download language tokenizer
    response = input("Do you want to download PaliGemma tokenizer? (y/n): ")
    if response.lower() == 'y':
        if not download_language_tokenizer():
            print("Warning: Failed to download PaliGemma tokenizer")
    
    # Setup ManiSkill assets
    print("\nSetting up ManiSkill assets...")
    if not setup_maniskill_assets():
        print("Failed to setup ManiSkill assets")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print("You can now run the evaluation script:")
    print("python scripts/eval_pi0_maniskill.py")
    print("\nOptional arguments:")
    print("  --env_id PutOnPlateInScene25VisionTexture03-v1")
    print("  --num_envs 10")
    print("  --num_episodes 20")
    print("  --device cuda:0")
    print("  --output_dir results/eval_pi0_maniskill")
    print("="*50)


if __name__ == "__main__":
    main() 