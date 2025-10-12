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
Test script to validate pi0 evaluation setup.
This script performs basic checks to ensure everything is configured correctly, including 
1. packages
2. CUDA availability
3. model files
4. pi0 model loading
5. ManiSkill environment creation
6. Language instruction
7. Environment reset

If the environment is setup correctly, the script will save a figure of the environment to the figs folder: scripts/setup/figs/env_id_xxx.jpg
"""

import sys
import torch
from pathlib import Path
from termcolor import colored  # Import termcolor for colored output
import argparse
import os
import numpy as np
from utils.custom_dirs import PI_R_ROOT_DIR
import matplotlib.pyplot as plt

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    required_imports = {
        "torch": "PyTorch",
        "transformers": "Transformers", 
        "gymnasium": "Gymnasium",
        "tyro": "Tyro",
        "numpy": "NumPy",
        "PIL": "Pillow",
    }
    
    failed_imports = []
    
    for module, name in required_imports.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {e}")
            failed_imports.append(name)
    
    # Test ManiSkill separately (may need special setup)
    try:
        import mani_skill
        print("✓ ManiSkill3")
    except ImportError as e:
        print(f"✗ ManiSkill3: {e}")
        failed_imports.append("ManiSkill3")
    
    return failed_imports


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
    else:
        print("⚠ CUDA not available - will use CPU (slower)")
    
    return torch.cuda.is_available()


def test_model_files():
    """Test if model files exist"""
    print("\nTesting model files...")
    
    # Check pi0 model
    pi0_path = Path(os.path.join(PI_R_ROOT_DIR, "physical-intelligence/pi0_base/pretrained_model"))
    if pi0_path.exists():
        print(f"✓ Pi0 model found at {pi0_path}")
        config_file = pi0_path / "config.json"
        model_file = pi0_path / "model.safetensors"
        
        if config_file.exists():
            print(f"  ✓ Config file found in {config_file}")
        else:
            print(f"  ✗ Config file missing, it should be placed in {config_file}")
            
        if model_file.exists():
            print(f"  ✓ Model weights found in {model_file}")
        else:
            print(f"  ✗ Model weights missing, it should be placed in {model_file}")
            
        return config_file.exists() and model_file.exists()
    else:
        print(f"✗ Pi0 model NOT found at {pi0_path}")
        print("  Run: python scripts/setup_pi0_eval.py")
        return False


def test_pi0_loading(model_device:str="cuda"):
    """Test loading the pi0 model"""
    print("\nTesting pi0 model loading...")
    
    try:
        # Add parent directory to path
        sys.path.append(str(Path(__file__).parent.parent))
        
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
        from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
        
        print("✓ Pi0 imports successful")
        
        # Test loading config
        model_path = os.path.join(PI_R_ROOT_DIR, "physical-intelligence/pi0_base/pretrained_model")
        if not Path(model_path).exists():
            print(f"✗ Model path does not exist: {model_path}")
            return False
        else:
            print(f"✓ Model path exists: {model_path}")
        
        config = PI0Config.from_pretrained(model_path)
        print("✓ Pi0 config loaded")
        
        # Test loading model (on CPU to avoid CUDA memory issues)
        config.device = model_device
        print(f"Loading pi0 model onto {model_device}")
        model = PI0Policy.from_pretrained(model_path, config=config)
        print(f"✓ Pi0 model loaded successfully onto {config.device}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load pi0 model: {e}")
        return False


def test_environment(sim_device:str="cuda",
                     num_envs:int=2,
                     env_id:str="PutOnPlateInScene25VisionTexture03-v1",
                     control_mode:str="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
                     obs_mode:str="rgb+segmentation",
                     max_episode_steps:int=10,
                     ):
    """Test ManiSkill environment creation."""
    print("\nTesting ManiSkill environment...")
    
    try:
        import gymnasium as gym
        print(f"Testing environment: {env_id}")

        # Use a context manager to temporarily set the CUDA device for simulation
        # This prevents changing the global default device for the rest of the script
        try:
            if sim_device.startswith("cuda"):
                sim_backend = "gpu"
                sim_device_id = int(sim_device.split(":")[1]) if ":" in sim_device else 0
                
                print(f"Attempting to set up simulation on CUDA device: {sim_device_id}")
                with torch.cuda.device(sim_device_id):
                    env = gym.make(
                        env_id,
                        num_envs=num_envs,
                        obs_mode=obs_mode,
                        control_mode=control_mode,
                        sim_backend=sim_backend,
                        max_episode_steps=max_episode_steps,
                    )
            else:
                sim_backend = "cpu"
                print(f"Using CPU backend for ManiSkill")
                env = gym.make(
                    env_id,
                    num_envs=num_envs,
                    obs_mode=obs_mode,
                    control_mode=control_mode,
                    sim_backend=sim_backend,
                    max_episode_steps=max_episode_steps,
                )

            print("✓ Environment created successfully")
            
            # Test reset
            obs, info = env.reset(seed=0)
            print(f"✓ Environment reset successful, obs={obs}, info={info}")
            env_test_imgs=obs['sensor_data']['3rd_view_camera']['rgb'] #[N_envs, H, W, 3]
            save_test_env_ims(env_test_imgs, output_dir=os.path.join(os.path.dirname(__file__), 'figs'), num_envs=num_envs, env_id=env_id)
            # Test language instruction
            try:
                instructions = env.unwrapped.get_language_instruction()
                print(f"✓ Language instructions: {instructions[:1]}...")
                has_language = True
            except Exception as e:
                print(f"✗ Language instruction failed: {e}")
                has_language = False
            
            env.close()
            return has_language
            
        except Exception as e:
            print(f"✗ Environment creation failed: {e}")
            return False
        
    except ImportError as e:
        print(f"✗ Failed to import ManiSkill: {e}")
        return False
    
    
    
def save_test_env_ims(rgb_data, output_dir, num_envs, env_id):
    # Extract RGB data # Shape: [N_envs, H, W, 3]

    # Convert to NumPy if it's a PyTorch tensor
    if isinstance(rgb_data, torch.Tensor):
        rgb_data = rgb_data.cpu().numpy()

    # Ensure data is uint8 and in range [0, 255]
    if rgb_data.dtype != np.uint8:
        if rgb_data.max() <= 1.0:
            rgb_data = (rgb_data * 255).astype(np.uint8)
        else:
            rgb_data = rgb_data.astype(np.uint8)

    # Create a grid of images
    # Determine grid size (e.g., try to make it as square as possible)
    cols = int(np.ceil(np.sqrt(num_envs)))
    rows = int(np.ceil(num_envs / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten() if num_envs > 1 else [axes]  # Handle single env case

    for i in range(num_envs):
        axes[i].imshow(rgb_data[i])  # Plot RGB image for env i
        axes[i].set_title(f"Env {i}")
        axes[i].axis('off')  # Hide axes for cleaner visualization

    # Turn off any unused subplots
    for i in range(num_envs, len(axes)):
        axes[i].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    image_path = os.path.join(output_dir, f"{env_id}_3rd_view.jpg")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(image_path, format='jpeg', bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory
    print(f"Saved combined figure to {image_path}")

def main():
    """Run all tests"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Pi0 ManiSkill3 Evaluation Setup Test')
    parser.add_argument('--model_device', type=str, default='cuda:0', help='Device for model (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--sim_device', type=str, default='cuda:0', help='Device for simulation (e.g., cuda:0, cuda:3, cpu)')
    parser.add_argument('--num_envs', type=int, default=2, help='Number of parallel environments')
    parser.add_argument('--env_id', type=str, default='PutOnPlateInScene25VisionTexture03-v1', help='Environment ID')
    parser.add_argument('--control_mode', type=str, default='arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos', help='Control mode for the environment')
    parser.add_argument('--obs_mode', type=str, default='rgb+segmentation', help='Observation mode (e.g., rgb+segmentation)')
    parser.add_argument('--max_episode_steps', type=int, default=500, help='Maximum steps per episode')
    args = parser.parse_args()

    print("Pi0 ManiSkill3 Evaluation Setup Test")
    print("=" * 50)
    
    # Assign arguments to variables
    model_device = args.model_device
    sim_device = args.sim_device
    num_envs = args.num_envs
    env_id = args.env_id
    control_mode = args.control_mode
    obs_mode = args.obs_mode
    max_episode_steps = args.max_episode_steps
    
    results = {}
    
    # # Test imports
    failed_imports = test_imports()
    results["imports"] = len(failed_imports) == 0
    
    # # # Test CUDA
    # results["cuda"] = test_cuda()
    
    # # Test model files
    # results["model_files"] = test_model_files()
    
    # # Test pi0 loading
    # results["pi0_loading"] = test_pi0_loading(model_device=model_device)
    
    # Test environment
    results["environment"] = test_environment(sim_device=sim_device,
                                              num_envs=num_envs,
                                              env_id=env_id,
                                              control_mode=control_mode,
                                              obs_mode=obs_mode,
                                              max_episode_steps=max_episode_steps)
    print(f"""results["environment"]={results["environment"]}""")
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else colored("✗ FAIL", "red")  # Color FAIL red
        print(f"{test_name.replace('_', ' ').title()}: {status}")
        if not passed:
            all_passed = False
    
    print("="*50)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("You can now run: python scripts/eval_pi0_maniskill.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please fix the issues above before running evaluation")
        
        if not results["imports"]:
            print("\n💡 Fix imports: python scripts/setup_pi0_eval.py")
        if not results["model_files"]:
            print("\n💡 Download models: python scripts/setup_pi0_eval.py")
        if not results["environment"]:
            print("\n💡 Check ManiSkill installation and assets")
    
    print("="*50)
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 