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


import os
import torch 
import torch.nn as nn
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.envs.sapien_env import BaseEnv
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
import numpy as np
import json
from typing import List
from mani_skill.utils.visualization.misc import images_to_video
from typing import Optional, Any, Dict
from logging import getLogger
logger=getLogger(__name__)

# Import tile_images for creating tiled videos
def apply_success_filter(img, success_status, filter_color=(0.7, 1.0, 0.7), alpha=0.3):
    """
    Apply a colored filter to images based on success status
    
    Args:
        img: Image array [H, W, C] with values in [0, 1] or [0, 255]
        success_status: Boolean indicating if this episode was successful
        filter_color: RGB color for the filter (default: light green)
        alpha: Transparency of the filter overlay
    
    Returns:
        Filtered image with same shape and dtype as input
    """
    if not success_status:
        return img
    
    # Ensure image is in [0, 1] range
    if img.max() > 1.0:
        img_normalized = img.astype(np.float32) / 255.0
        was_uint8 = True
    else:
        img_normalized = img.astype(np.float32)
        was_uint8 = False
    
    # Create filter overlay
    filter_overlay = np.ones_like(img_normalized) * np.array(filter_color)
    
    # Blend image with filter
    filtered_img = (1 - alpha) * img_normalized + alpha * filter_overlay
    filtered_img = np.clip(filtered_img, 0, 1)
    
    # Convert back to original dtype
    if was_uint8:
        filtered_img = (filtered_img * 255).astype(np.uint8)
    
    return filtered_img

def tile_images(img_nhwc, success_flags=None):
    """
    Tile N images into one big PxQ image with optional success filtering
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    success_flags: Optional list of boolean flags indicating success for each environment
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    
    # Apply success filter if flags are provided
    if success_flags is not None:
        for i in range(N):
            if i < len(success_flags):
                img_nhwc[i] = apply_success_filter(img_nhwc[i], success_flags[i])
    
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c

def save_config(cfg, log_dir: Path, cfg_name:str='cfg'):
    if cfg_name=='cfg_model':
        print(f"Debug:: save_config: cfg.num_steps={cfg.num_steps}")
    """Save the evaluation configuration to log_dir/cfg.yaml"""
    config_path = os.path.join(log_dir,f"{cfg_name}.yaml")
    with open(config_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    print(f"Configuration saved to: {config_path}")

def save_model_architecture(model:nn.Module, log_dir: Path):
    """Save the model architecture to log_dir/architecture.log"""
    if isinstance(log_dir, str):
        log_dir=Path(log_dir)
    arch_path = log_dir / "architecture.log"
    
    # Capture model architecture information
    arch_info = []
    arch_info.append("=" * 50)
    arch_info.append("PI0 MODEL ARCHITECTURE")
    arch_info.append("=" * 50)
    arch_info.append("")
    
    # Model summary
    arch_info.append(f"\nModel Type: {type(model).__name__}")
    arch_info.append(f"\nDevice: {next(model.parameters()).device}")
    arch_info.append(f"\n{model.__class__.__name__}=={model}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    arch_info.append(f"\nTotal Parameters: {total_params:,}\n")
    arch_info.append(f"\nTrainable Parameters: {trainable_params:,}\n")
    arch_info.append("\n")
    
    # Model structure
    arch_info.append("\nModel Structure:")
    arch_info.append("-" * 30)
    
    def add_module_info(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            arch_info.append(f"{full_name}: {type(child).__name__}")
            if hasattr(child, 'in_features') and hasattr(child, 'out_features'):
                arch_info.append(f"  Input features: {child.in_features}")
                arch_info.append(f"  Output features: {child.out_features}")
            elif hasattr(child, 'num_heads'):
                arch_info.append(f"  Num heads: {child.num_heads}")
            elif hasattr(child, 'hidden_size'):
                arch_info.append(f"  Hidden size: {child.hidden_size}")
            add_module_info(child, full_name)
    
    add_module_info(model)
    
    # Save to file
    with open(arch_path, 'w') as f:
        f.write(f"Model Type: {type(model).__name__}\n")
        f.write(f"Device: {next(model.parameters()).device}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"{model.__class__.__name__}=={model}")
        f.write('\n'.join(arch_info))
    print(f"Model architecture saved to: {arch_path}")


def eval_prompt(cfg_eval):
    print("Pi0 ManiSkill3 Evaluation")
    print("=" * 50)
    print(f"Environment: {cfg_eval.env.id}")
    print(f"Model: {cfg_eval.model.path}")
    print(f"Device (model): {cfg_eval.model.device}")
    print(f"Device (sim): {cfg_eval.sim.device}")
    print(f"Output dir: {cfg_eval.output.dir}")
    print(f"Seed: {cfg_eval.eval.seed}")
    print("=" * 50)

def make_logging_dirs(output_dir):
    # Create output directories
    output_dir = Path(output_dir)
    video_dir = output_dir / "videos"
    data_dir = output_dir / "data"
    log_dir = output_dir / "logs"
    for dir_path in [output_dir, video_dir, data_dir, log_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    return video_dir, data_dir, log_dir


def set_model_sim_devices(cfg):
    print(f"=== DEVICE SETUP DEBUG ===")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"Requested model device: {cfg.model.device}")
    print(f"Requested sim device: {cfg.sim.device}")
    
    if torch.cuda.is_available():
        print(f"Available CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  CUDA device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        
        # Try to show physical GPU mapping
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            print(f"Physical GPUs available: {device_count}")
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                print(f"  Physical GPU {i}: {name}")
        except:
            print("Could not get physical GPU mapping (pynvml not available)")

    # Determine model device
    if "cuda" in cfg.model.device and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        cfg.model.device = "cpu"
    if cfg.model.device.startswith("cuda"):
        device_id = int(cfg.model.device.split(':')[-1])
        print(f"Model device requested ID: {device_id}")
        if device_id >= torch.cuda.device_count():
            print(f"Warning: CUDA device {device_id} not available. Available devices: 0-{torch.cuda.device_count()-1}")
            print("Falling back to cuda:0")
            cfg.model.device = "cuda:0"
            device_id = 0
        print(f"Model device final ID: {device_id}")
    model_device = torch.device(cfg.model.device)
    model_device_id=int(cfg.model.device.split(':')[-1]) if cfg.model.device.startswith('cuda') else None
    # Print device information for debugging
    print(f"Model device: {model_device}")
    print(f"Model device ID: {model_device_id}")
    
    # Determine simulation backend and device management
    sim_device = torch.device(cfg.sim.device) if cfg.sim.device.startswith("cuda") else torch.device("cpu")
    sim_backend = "gpu" if cfg.sim.device.startswith("cuda") else "cpu"
    sim_device_id = int(cfg.sim.device.split(":")[1]) if ":" in cfg.sim.device else 0
    print(f"Sim device: {sim_device}")
    print(f"Sim device ID: {sim_device_id}")
    print(f"Sim backend: {sim_backend}")
    
    # Test actual device allocation
    if torch.cuda.is_available():
        print(f"=== TESTING DEVICE ALLOCATION ===")
        try:
            # Test model device
            test_tensor_model = torch.tensor([1.0]).to(model_device)
            print(f"Model device test successful: tensor on {test_tensor_model.device}")
            
            # Test sim device  
            test_tensor_sim = torch.tensor([1.0]).to(sim_device)
            print(f"Sim device test successful: tensor on {test_tensor_sim.device}")
            
            # Show current device after allocation
            print(f"Current CUDA device after allocation: {torch.cuda.current_device()}")
            
        except Exception as e:
            print(f"Device allocation test failed: {e}")
    
    # Check if model and sim are on different devices
    if sim_device != model_device:
        print(f"Multi-device setup: Model on {model_device}, Simulation on {sim_device}")
        print("Tensors will be transferred between devices as needed.")
    
    print(f"=== END DEVICE SETUP DEBUG ===")
    return sim_backend, sim_device, sim_device_id, model_device


def load_pi0_model(model_path, model_device, dataset_stats_path, model_overrides:dict)->PI0Policy:
    print(f"Loading dataset statistics from dataset_stats_filename= {dataset_stats_path}")
    
    # check available device
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Model device requested: {model_device}")
    print(f"Model device type: {type(model_device)}")
    # Check if the requested device is valid
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    
    # Load dataset statistics
    dataset_stats=torch.load(dataset_stats_path)
    if dataset_stats is None:
        raise ValueError(f"Your dataset statistics is empty: {dataset_stats_path}, maybe checkout what is in that file?")
    print(f"Successfully loaded dataset_stats:{type(dataset_stats)}. \nContent: {dataset_stats}")
    
    # Load pi0 model
    print(f"Trying to load pi0 model from {model_path}...")
    try:
        cfg_model: PI0Config = PI0Config.from_pretrained(model_path) # this will automatically load the configs from cfg.model.path/config.json if the path exists. 
        # Override command-prompt input attributes from the configurable cfg.yaml file to the cfg_model loaded from the static config.json file in the checkpoint directory.
        if model_overrides:
            logger.info(f"Applying config overrides {model_overrides} to the model file...")
            for key, value in model_overrides.items():
                if value is not None:  # Skip null values to use original config
                    if hasattr(cfg_model, key):
                        original_value = getattr(cfg_model, key)
                        setattr(cfg_model, key, value)
                        logger.info(f"Override {key}: {original_value} -> {value}")
                    else:
                        logger.warning(f"Config key '{key}' not found in model config, skipping override")
                else:
                    logger.info(f"Skipping override for {key} (value is null)")
        # Use string representation of device for config
        cfg_model.device = str(model_device)
        
        # Finally, load the model from the checkpoint directory according to the overridden cfg_model.
        model = PI0Policy.from_pretrained(pretrained_name_or_path=model_path, 
                                          config=cfg_model, 
                                          dataset_stats=dataset_stats)
        model.to(model_device)
        model.eval()
        print(f"num_steps={model.config.num_steps}")
        print(f"cfg_model.num_steps={cfg_model.num_steps}")
        print(f"Successfully loaded pi0 model on {model_device}")
    except Exception as e:
        print(f"Error loading pi0 model: {e}")
        print("Make sure you have downloaded the pi0 model weights.")
        raise
    print(f"Successfully loaded pi0 model on {model_device} ")
    return model



def reset_env_model(env:BaseEnv, model:PreTrainedPolicy, base_seed:int, use_different_seeds:bool=False, reset_model:bool=True):
    # Reset environment and model with seeds. 
    if not use_different_seeds:
        seeds=base_seed   # use the same seed for all the parallel sthe environment. 
    else:
        seeds=[base_seed+i for i in range(env.num_envs)]  # use different seeds for different parallel environments.
     
    # leave options as blank, so that the env will automatically call randomization scripts like 
    # `_initialize_episode_pre` to set different initial objects, tables, overlays, etc. to different environments. 
    # warning: you should not manually set episode_idx as some small numbers, this will cause 
    # the objects and tables to fall into the same category, thus cancelling the automatic randomization.     
    opts={}
    print(f"Resetting environment with seed: {seeds}, options: {opts}")
    
    obs, step_info = env.reset(seed=seeds, options=opts) # type: ignore
    obs: dict
    
    # Reset model: Clears conversation history, Resets attention mechanisms, Clears any cached activations
    if reset_model and model:
        model.reset()
    
    return obs, step_info

                

def analyze_space(env, action_space):
    """For debug use. Analyze and print detailed information about the action space, robot state, and the controller"""
    print("\n" + "="*60)
    print("DETAILED ACTION SPACE ANALYSIS")
    print("="*60)
    
    # Basic action space info
    print(f"Action space type: {type(action_space).__name__}")
    print(f"Action space shape: {action_space.shape}")
    print(f"Action space dtype: {action_space.dtype}")
    print(f"Action space low: {action_space.low}")
    print(f"Action space high: {action_space.high}")
    
    # Robot and controller info
    robot = env.agent.robot
    controller = env.agent.controller
    
    print(f"\nRobot Information:")
    print(f"  Robot type: {type(robot).__name__}")
    print(f"  Robot DOF: {robot.max_dof}")
    print(f"  Control mode: {env.control_mode}")
    
    print(f"\nController Information:")
    print(f"  Controller type: {type(controller).__name__}")
    print(f"  Controller action space: {controller.action_space}")
    
    # Current robot state
    qpos = robot.get_qpos()
    qvel = robot.get_qvel()
    print(f"\nCurrent Robot State:")
    print(f"  Joint positions (qpos): {qpos.shape}, {qpos}")
    print(f"  Joint velocities (qvel): {qvel.shape}, {qvel}")
    
    # Try to get end-effector information
    try:
        if hasattr(robot, 'get_ee_pose'):
            ee_pose = robot.get_ee_pose()
            print(f"  End-effector pose: {ee_pose}")
    except:
        print("  End-effector pose: Not available")
    
    # Action space interpretation
    print(f"\nAction Space Interpretation:")
    if action_space.shape[0] == 7:
        print("  This is likely a 7-DOF robot with end-effector control:")
        print("  Dimensions 0-2: End-effector position (x, y, z) in meters")
        print("  Dimensions 3-5: End-effector orientation (roll, pitch, yaw) in radians")
        print("  Dimension 6: Gripper action (open/close)")
        
        print(f"\nValue Ranges:")
        print(f"  Position (x,y,z): [{action_space.low[0]:.3f}, {action_space.high[0]:.3f}] meters")
        print(f"  Orientation (r,p,y): [{action_space.low[3]:.3f}, {action_space.high[3]:.3f}] radians (±{np.degrees(action_space.high[3]):.1f}°)")
        print(f"  Gripper: [{action_space.low[6]:.3f}, {action_space.high[6]:.3f}]")
        
        # Show example actions
        print(f"\nExample Actions:")
        print(f"  Zero action (no movement): {np.zeros(7)}")
        print(f"  Max position forward: {[action_space.high[0], 0, 0, 0, 0, 0, 0]}")
        print(f"  Max rotation: {[0, 0, 0, action_space.high[3], 0, 0, 0]}")
        print(f"  Open gripper: {[0, 0, 0, 0, 0, 0, action_space.low[6]]}")
        print(f"  Close gripper: {[0, 0, 0, 0, 0, 0, action_space.high[6]]}")
    
    elif action_space.shape[0] == robot.max_dof:
        print(f"  This appears to be joint space control with {robot.max_dof} DOF")
        print("  Each dimension corresponds to a joint position/velocity")
    
    else:
        print(f"  Unknown action space configuration with {action_space.shape[0]} dimensions")
    
    print("="*60)
    return action_space


# New batch-based data recording functions
class BatchEpisodeData:
    """Efficient batch-based episode data storage designed for parallel simulation. """
    
    def __init__(self, num_envs: int, instructions: List[str]):
        self.num_envs = num_envs
        self.tiled_images = []  # Store tiled images (all envs in one frame)
        self.tiled_images_success_filtered = []  # Store tiled images with success filter applied
        self.individual_images = []  # Store individual environment images [step][env_idx] = image
        self.rewards = [[] for _ in range(num_envs)]
        self.infos = [[] for _ in range(num_envs)]
        self.actions = [[] for _ in range(num_envs)]  # Store actions for each environment
        self.instructions = instructions
        self.success = [False] * num_envs
        self.step_success = []  # Track success status at each step for all environments
        
    def add_step_data_in_batch(self, obs_rgb_batch: torch.Tensor, rewards: torch.Tensor, step_info: dict, actions_batch: torch.Tensor|None = None):
        """Record step data for all environments in batch"""
        # Convert batch images to CPU and numpy
        if obs_rgb_batch.device != torch.device("cpu"):
            obs_rgb_batch = obs_rgb_batch.cpu()
        
        # obs_rgb_batch is already in [B, H, W, C] format from simulator
        images_bhwc = obs_rgb_batch.numpy()
        
        # Store individual images for later use
        self.individual_images.append([images_bhwc[i] for i in range(self.num_envs)])
        
        # Get current success status for each environment
        current_success_flags = []
        for env_idx in range(self.num_envs):
            # Check if this environment has achieved success so far
            env_success = False
            if self.infos[env_idx]:  # If we have previous step info
                env_success = any(info.get("success", False) for info in self.infos[env_idx])
            # Also check current step
            if hasattr(step_info.get('success', []), '__getitem__') and len(step_info['success']) > env_idx:
                env_success = env_success or step_info['success'][env_idx]
            current_success_flags.append(env_success)
        
        # Store step success status
        self.step_success.append(current_success_flags.copy())
        
        # Create regular tiled image (all environments in one frame)
        tiled_image = tile_images(images_bhwc)
        self.tiled_images.append(tiled_image)
        
        # Create success-filtered tiled image
        tiled_image_filtered = tile_images(images_bhwc, success_flags=current_success_flags)
        self.tiled_images_success_filtered.append(tiled_image_filtered)
        
        # Store rewards, actions and step info for all environments
        for env_idx in range(self.num_envs):
            # Store rewards
            reward_val = rewards[env_idx].item() if hasattr(rewards[env_idx], 'item') else rewards[env_idx]
            self.rewards[env_idx].append(reward_val)
            
            # Store actions - ensure JSON serializable
            if actions_batch is not None:
                action_val = actions_batch[env_idx]
                
                # Convert to CPU if on CUDA
                if hasattr(action_val, 'cpu'):
                    action_val = action_val.cpu()
                
                # Convert torch tensor to numpy then to list
                if hasattr(action_val, 'numpy'):
                    action_val = action_val.numpy().tolist()
                elif hasattr(action_val, 'tolist'):
                    action_val = action_val.tolist()
                elif isinstance(action_val, np.ndarray):
                    action_val = action_val.tolist()
                
                self.actions[env_idx].append(action_val)
            
            # Store step info
            env_info = {}
            for key, value in step_info.items():
                if hasattr(value, '__getitem__') and len(value) > env_idx:
                    env_info[key] = (
                        value[env_idx].item() 
                        if hasattr(value[env_idx], 'item') 
                        else value[env_idx]
                    )
                else:
                    env_info[key] = value
            self.infos[env_idx].append(env_info)
    
    def add_final_observation_in_batch(self, obs_rgb_batch: torch.Tensor):
        """Add final observation for all environments"""
        # Convert batch images to CPU and numpy
        if obs_rgb_batch.device != torch.device("cpu"):
            obs_rgb_batch = obs_rgb_batch.cpu()
            
        # obs_rgb_batch is already in [B, H, W, C] format from simulator
        images_bhwc = obs_rgb_batch.numpy()
        
        # Store final individual images
        self.individual_images.append([images_bhwc[i] for i in range(self.num_envs)])
        
        # Determine final success status (before finalize_success_status is called)
        final_success_flags = []
        for env_idx in range(self.num_envs):
            env_success = False
            if self.infos[env_idx]:  # Check if any step was successful
                env_success = any(info.get("success", False) for info in self.infos[env_idx])
            final_success_flags.append(env_success)
        
        # Store final step success status
        self.step_success.append(final_success_flags.copy())
        
        # Create final tiled image
        tiled_image = tile_images(images_bhwc)
        self.tiled_images.append(tiled_image)
        
        # Create final success-filtered tiled image
        tiled_image_filtered = tile_images(images_bhwc, success_flags=final_success_flags)
        self.tiled_images_success_filtered.append(tiled_image_filtered)
    
    def finalize_success_status(self):
        """Determine success status for each environment in this episode"""
        for env_idx in range(self.num_envs):
            if self.infos[env_idx]:
                self.success[env_idx] = any(
                    info.get("success", False) for info in self.infos[env_idx]
                )
    
    def get_tiled_video_data(self) -> dict:
        """Get tiled video data (all environments in one video)"""
        return {
            "images": self.tiled_images,
            "success_rate": np.mean(self.success),
            "num_envs": self.num_envs,
            "instructions": self.instructions
        }
    
    def get_tiled_video_data_with_success_filter(self) -> dict:
        """Get tiled video data with success highlighting (all environments in one video)"""
        return {
            "images": self.tiled_images_success_filtered,
            "success_rate": np.mean(self.success),
            "num_envs": self.num_envs,
            "instructions": self.instructions,
            "type": "success_filtered"
        }
    
    def get_success_only_video_data(self) -> dict:
        """Get video data for only successful environments"""
        successful_env_indices = [i for i, success in enumerate(self.success) if success]
        
        if not successful_env_indices:
            return {
                "images": [],
                "success_rate": 0.0,
                "num_envs": 0,
                "instructions": [],
                "message": "No successful episodes to create video"
            }
        
        # Create tiled images with only successful environments
        success_only_images = []
        for step_idx in range(len(self.individual_images)):
            # Get images from successful environments only
            step_images = []
            for env_idx in successful_env_indices:
                if env_idx < len(self.individual_images[step_idx]):
                    step_images.append(self.individual_images[step_idx][env_idx])
            
            if step_images:
                # Tile only successful environment images
                step_images_array = np.array(step_images)
                # Apply green filter to all (since they're all successful)
                success_flags = [True] * len(step_images)
                tiled_success_image = tile_images(step_images_array, success_flags=success_flags)
                success_only_images.append(tiled_success_image)
        
        return {
            "images": success_only_images,
            "success_rate": 1.0,  # 100% since we only include successful ones
            "num_envs": len(successful_env_indices),
            "successful_env_indices": successful_env_indices,
            "instructions": [self.instructions[i] for i in successful_env_indices],
            "type": "success_only"
        }

    def get_env_data(self, env_idx: int) -> dict:
        """Get data for a specific environment (for backward compatibility, actually not used anywhere currently.)"""
        return {
            "rewards": self.rewards[env_idx],
            "actions": self.actions[env_idx],
            "info": self.infos[env_idx],
            "instruction": self.instructions[env_idx],
            "success": self.success[env_idx]
        }
    

def create_batch_episode_data(num_envs: int, instructions: List[str]) -> BatchEpisodeData:
    """Create a new BatchEpisodeData instance"""
    return BatchEpisodeData(num_envs, instructions)


def _compute_action_statistics(all_actions: List[List]) -> dict:
    """Compute basic statistics for recorded actions"""
    if not all_actions or not any(all_actions):
        return {"message": "No actions recorded"}
    
    # Flatten all actions across environments and steps
    flattened_actions = []
    for env_actions in all_actions:
        for action in env_actions:
            if action is not None and action != []:
                # Actions are now stored as lists, so they should be ready for numpy
                flattened_actions.append(action)
    
    if not flattened_actions:
        return {"message": "No valid actions found"}
    
    # Convert to numpy array for statistics
    try:
        actions_array = np.array(flattened_actions)  # [total_steps, action_dim]
        
        # Handle case where actions might be scalars
        if len(actions_array.shape) == 1:
            actions_array = actions_array.reshape(-1, 1)
        
        action_stats = {
            "total_actions": len(flattened_actions),
            "action_dim": actions_array.shape[1] if len(actions_array.shape) > 1 else 1,
            "mean": actions_array.mean(axis=0).tolist(),
            "std": actions_array.std(axis=0).tolist(),
            "min": actions_array.min(axis=0).tolist(),
            "max": actions_array.max(axis=0).tolist(),
            "shape": list(actions_array.shape)
        }
        
        # Add per-dimension statistics
        if len(actions_array.shape) > 1 and actions_array.shape[1] <= 10:  # Only for reasonable action dimensions
            action_stats["per_dimension"] = {}
            for dim in range(actions_array.shape[1]):
                action_stats["per_dimension"][f"dim_{dim}"] = {
                    "mean": float(actions_array[:, dim].mean()),
                    "std": float(actions_array[:, dim].std()),
                    "min": float(actions_array[:, dim].min()),
                    "max": float(actions_array[:, dim].max())
                }
        
        return action_stats
        
    except Exception as e:
        return {"error": f"Failed to compute action statistics: {str(e)}", "debug_info": {
            "num_flattened_actions": len(flattened_actions),
            "first_action_type": str(type(flattened_actions[0])) if flattened_actions else "None",
            "first_action_shape": str(np.array(flattened_actions[0]).shape) if flattened_actions else "None"
        }}


def save_batch_episode_data(batch_data: BatchEpisodeData, data_dir: Path):
    """Save episode data efficiently - all environments together"""
    import json
    
    def make_json_serializable(obj):
        """Recursively convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        else:
            return obj
    
    # Save comprehensive batch data
    batch_summary = {
        "num_envs": batch_data.num_envs,
        "success_rate": float(np.mean(batch_data.success)),  # Ensure float, not numpy float
        "individual_success": batch_data.success,
        "instructions": batch_data.instructions,
        "episode_lengths": [len(infos) for infos in batch_data.infos],
        "all_rewards": batch_data.rewards,  # [env_idx][step] = reward
        "all_actions": batch_data.actions,  # [env_idx][step] = action_list (already converted to lists)
        "all_infos": batch_data.infos,      # [env_idx][step] = info_dict
        "total_steps": len(batch_data.tiled_images), # across all environments
        "action_statistics": _compute_action_statistics(batch_data.actions)
    }
    
    # Ensure all data is JSON serializable (safety check)
    batch_summary = make_json_serializable(batch_summary)
    
    # Save all data in one file
    summary_filepath = data_dir / "batch_episode_data.json"
    with open(summary_filepath, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    action_count = sum(len(env_actions) for env_actions in batch_data.actions)
    print(f"Saved batch episode data for {batch_data.num_envs} environments to {summary_filepath}")
    print(f"  - Recorded {action_count} total actions across all environments and steps")
    print(f"  - Action statistics included in the saved data")


def create_batch_videos(batch_data: BatchEpisodeData, video_dir: Path, fps: int = 10):
    """Create videos efficiently - regular, success-filtered, and success-only videos"""
    if isinstance(video_dir, str):
        video_dir = Path(video_dir)
    print(f"\nCreating evaluation videos...")
    
    success_rate = np.mean(batch_data.success)
    num_successful = sum(batch_data.success)
    
    print(f"Creating videos: {batch_data.num_envs} total environments, {num_successful} successful ({success_rate:.2%})")
    
    # 1. Create regular tiled video (all environments)
    tiled_video_data = batch_data.get_tiled_video_data()
    filename_regular = f"tiled_all_envs_success_rate_{success_rate:.2f}"
    
    if tiled_video_data["images"]:
        images_to_video(
            tiled_video_data["images"],
            str(video_dir),
            filename_regular,
            fps=fps,
            verbose=False
        )
        print(f"  ✓ Regular video saved: {video_dir/filename_regular}.mp4")
    
    # 2. Create success-filtered tiled video (all environments with green filter)
    success_filtered_data = batch_data.get_tiled_video_data_with_success_filter()
    filename_filtered = f"tiled_all_envs_with_success_filter_success_rate_{success_rate:.2f}"
    
    if success_filtered_data["images"]:
        images_to_video(
            success_filtered_data["images"],
            str(video_dir),
            filename_filtered,
            fps=fps,
            verbose=False
        )
        print(f"  ✓ Success-filtered video saved: {video_dir/filename_filtered}.mp4")
    
    # 3. Create success-only tiled video (only successful environments)
    if num_successful > 0:
        success_only_data = batch_data.get_success_only_video_data()
        filename_success_only = f"tiled_success_only_{num_successful}_envs"
        
        if success_only_data["images"] and "message" not in success_only_data:
            images_to_video(
                success_only_data["images"],
                str(video_dir),
                filename_success_only,
                fps=fps,
                verbose=False
            )
            successful_indices = success_only_data["successful_env_indices"]
            print(f"  ✓ Success-only video saved: {video_dir/filename_success_only}.mp4 (environments: {successful_indices})")
        else:
            print(f"  ⚠ No successful episodes - skipping success-only video")
    else:
        print(f"  ⚠ No successful episodes - skipping success-only video")
    print(f"Videos creation completed!")


def print_batch_step_info(step: int, batch_data: BatchEpisodeData):
    """Print step info for batch data"""
    if step % 10 == 0:
        current_success_rate = np.mean([
            any(info.get("success", 0) for info in env_infos) > 0 
            for env_infos in batch_data.infos
        ])
        print(f"Step {step}: Current Success Rate = {current_success_rate*100:.2f}% ({batch_data.num_envs} envs)")

# Keep old functions for backward compatibility if needed elsewhere
def record_episode_step_data(episode_data, obs_rgb, reward, step_info):
    for env_idx in range(len(episode_data)):
        # Get image from simulation device and transfer to CPU for saving
        img = obs_rgb[env_idx]    # [C, H, W]
        if img.device != torch.device("cpu"):
            img = img.cpu()
        episode_data[env_idx]["images"].append(img.numpy())
        # Store rewards
        episode_data[env_idx]["rewards"].append(reward[env_idx].item() if hasattr(reward, '__getitem__') else 0.0)
        # Store info: first convert to serializable format, then save
        env_info = {}
        for key, value in step_info.items():
            if hasattr(value, '__getitem__') and len(value) > env_idx:
                env_info[key] = (
                    value[env_idx].item() 
                    if hasattr(value[env_idx], 'item') 
                    else value[env_idx]
                )
            else:
                env_info[key] = value
        episode_data[env_idx]["info"].append(env_info)
    return episode_data

def record_episode_final_data(episode_data, obs_rgb):
    for env_idx in range(len(episode_data)):
        # Get final image and transfer to CPU for saving
        final_image = obs_rgb[env_idx]
        if final_image.device != torch.device("cpu"):
            final_image = final_image.cpu()
        episode_data[env_idx]["images"].append(final_image.numpy())
        
        # Determine success in this episode
        if episode_data[env_idx]["info"]:
            episode_data[env_idx]["success"] = any(info.get("success", False) for info in episode_data[env_idx]["info"])
                    
def prt_step_info(step, episode_data):
    # Print step info
    if step % 10 == 0:
        success_counts = [
            sum(info.get("success", 0) for info in data["info"]) > 0 
            for data in episode_data
        ]
        success_rate = np.mean(success_counts)
        print(f"Step {step}: Current Success Rate = {success_rate*100:.2f}%")



def log_step(action, ret_batch, prt_next_obs=True):
    next_obs, reward, terminated, truncated, step_info=ret_batch
    print(f"terminated={terminated.shape}, {terminated}")
    print(f"truncated={truncated.shape}, {truncated}")
    if prt_next_obs:
        print(f"next_obs={type(next_obs)}, {next_obs.keys()}") # {next_obs}
    print(f"reward={reward}")  # it is always a zero tensor of shape [num_envs,]
    print(f"step_info={step_info}") # step_info['success'] indicates episodes success. other keys include: elapsed_steps, is_src_obj_grasped, ...
    print(f"step_info['success']={step_info['success']}")
    print(f"action={action.shape}")  # Log the action that was taken   , {action}

def log(env_id, model_path, num_envs, base_seed, eval_metrics, log_dir):
    # Save final results
    results = {
        "config": {
            "env_id": env_id,
            "model_path": model_path,
            "num_envs": num_envs,
            "seed": base_seed,
        },
        "metrics": {},
        "raw_data": dict(eval_metrics)
    }
    # Calculate summary statistics
    for metric, values in eval_metrics.items():
        if values:
            results["metrics"][metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "count": len(values)
            }
    # Save results
    results_file = log_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY")
    print("="*50)
    print(f"Environment: {env_id}")
    print(f"Episodes evaluated: {len(eval_metrics.get('success', []))}")
    
    if 'success' in results["metrics"]:
        success_rate = results["metrics"]["success"]["mean"] * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    if 'episode_length' in results["metrics"]:
        avg_length = results["metrics"]["episode_length"]["mean"]
        print(f"Average episode length: {avg_length:.1f} steps")
    
    return results["metrics"]["success"]["mean"]

def load_eval_config(eval_config_path: str = "evaluate/config/default.yaml", 
                     overrides: Optional[Dict[str, Any]] = None) -> DictConfig:
    """Load evaluation configuration from yaml file with optional overrides"""
    config_path = Path(__file__).parent.parent / eval_config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Evaluation config file not found: {config_path}")
    
    # Load the config using OmegaConf
    cfg_eval = OmegaConf.load(config_path)
    
    # Ensure it's a DictConfig (not ListConfig)
    if not isinstance(cfg_eval, DictConfig):
        raise ValueError(f"Expected DictConfig but got {type(cfg_eval)}")
    
    # Apply overrides if provided
    if overrides:
        merged_cfg = OmegaConf.merge(cfg_eval, overrides)
        # Ensure the merged result is still a DictConfig
        if not isinstance(merged_cfg, DictConfig):
            raise ValueError(f"Config merge resulted in {type(merged_cfg)} instead of DictConfig")
        cfg_eval = merged_cfg
        logger.info(f"Applied config overrides: {overrides}")
    
    logger.info(f"Loaded evaluation config from: {config_path}")
    return cfg_eval

def load_eval_config_with_overrides(
    eval_config_path: str = "evaluate/config/default.yaml",
    model_path: Optional[str] = None,
    model_device: Optional[str] = None, 
    sim_device: Optional[str] = None, 
) -> DictConfig:
    """Load evaluation config with common overrides"""
    
    # Build overrides dictionary
    overrides = {}
    if model_path:
        overrides["model.path"] = model_path
    if model_device:
        overrides["model.device"] = model_device
    if sim_device:
        overrides["sim.device"] = sim_device
    
    return load_eval_config(eval_config_path, overrides)

