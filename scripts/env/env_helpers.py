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





import torch 
from mani_skill.envs.sapien_env import BaseEnv
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
from gymnasium.core import Env
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import logging
logger=logging.getLogger(__name__)

def setup_maniskill_env(env_id, 
                        num_envs, 
                        max_episode_len,
                        sim_backend, 
                        sim_device, 
                        sim_device_id,
                        sim_config,
                        sensor_config,
                        obs_mode,
                        control_mode,
                        episode_mode='eval',
                        wrappers=None, # list of wrappers to wrap the environment. 
                        )->ManiSkillVectorEnv:
    """
    mode:   'eval':
        # Parallel evaluation, run only one episode per environment, no reset within. 
            'train':
        # Parallel RL training, run multiple episodes per environment, each environment automatically resets and does not interfere with others. 
    """
    # Setup ManiSkill3 environment
    logger.info(f"Setting up ManiSkill3 environment: {env_id}")
    
    # Convert OmegaConf objects to regular Python dictionaries to avoid compatibility issues
    if isinstance(sim_config, DictConfig):  
        sim_config = OmegaConf.to_container(sim_config, resolve=True)
    if isinstance(sensor_config, DictConfig):
        sensor_config = OmegaConf.to_container(sensor_config, resolve=True)
    
    # Determine autoreset. 
    if episode_mode=='eval':
        # Parallel evaluation, run only one episode per environment, no reset within. 
        auto_reset=False
        ignore_terminations=True
        reconfiguration_freq=1
    elif episode_mode=='train':
        # Parallel RL training, run multiple episodes per environment, each environment automatically resets and does not interfere with others. 
        auto_reset=True
        ignore_terminations=False
        reconfiguration_freq=0
    else:
        raise NotImplementedError(f"Episode mode {episode_mode} not implemened, we only support 'eval' and 'train'.")
    try:
        if sim_backend == "gpu":
            logger.info(f"Setting up simulation on CUDA device: {sim_device}")
            logger.info(f"About to call torch.cuda.set_device({sim_device_id}). Current CUDA device before set_device: {torch.cuda.current_device()}")
            
            # Set the CUDA device for the entire environment creation process
            torch.cuda.set_device(sim_device_id)
            
            logger.info(f"Current CUDA device after set_device: {torch.cuda.current_device()}")
            logger.info(f"Available devices: {torch.cuda.device_count()}")
            
            env: BaseEnv = gym.make( # type: ignore
                id=env_id,
                num_envs=num_envs,
                max_episode_steps=max_episode_len,
                obs_mode=obs_mode,
                # reward_mode='sparse',
                control_mode=control_mode,
                sim_backend=sim_backend,
                # render_backend=sim_backend,
                sim_config=sim_config,
                sensor_configs=sensor_config,
                reconfiguration_freq=reconfiguration_freq
            )# type: ignore
            
            # Wrap the environment with the wrappers, if provided. 
            logger.info(f"$ -------------------------------------------------------------------------- $")
            if wrappers:
                logger.info(f"Wrapping up BaseEnv with custom wrappers:")
                for wrapper in wrappers:
                    logger.info(f"\tWrapping up BaseEnv with custom wrapper {wrapper.func.__name__}")
                    env = wrapper(env) 
            logger.info(f"$ -------------------------------------------------------------------------- $")
            logger.info(f"Wrapping environment with ManiSkillVectorEnv with auto_reset={auto_reset} and ignore_terminations={ignore_terminations}.")
            venv: ManiSkillVectorEnv=ManiSkillVectorEnv(env,
                                                        auto_reset= auto_reset,
                                                        ignore_terminations= ignore_terminations)
            logger.info(f"Double checking: env.base_env.reconfiguration_freq={venv.base_env.reconfiguration_freq}, env.auto_reset={venv.auto_reset} env.ignore_terminations={venv.ignore_terminations}")
            logger.info(f"$ -------------------------------------------------------------------------- $")
        else:
            raise NotImplementedError
    except Exception as e:
        logger.info(f"Error setting up ManiSkill environment: {e}")
        raise
    logger.info(f"Successfully setup ManiSkill3 environment {env} on sim_device={sim_device}")
    
    env_unwrapped: Env=venv.unwrapped
    logger.info(f"Sim environment details:")
    env_unwrapped.print_sim_details()
    return venv # type: ignore


def fetch_rgb_from_obs(obs:dict, sim_device, model_device):
    """
    This function extracts the rgb component from the full raw observation from ManiSkill3-simpler-env
    and converts the the shape, dtype and device that is compatible with pi-zero model's inputs. 
    Returns:
        img_list: List[torch.Tensor]
        img_list[0]: [B,C,H,W]
        img_list[1]: [B,C,H,W]
        ...
        img_list[n_cameras-1]: [B,C,H,W]
    """
    if not isinstance(obs, dict):
        raise ValueError(f"obs is {type(obs)} but not dict. check your code. obs={obs}")
    n_cameras=1
    img_list = [None] * n_cameras  # Initialize list with None values to be filled
    for camera_id in range(n_cameras):
        if camera_id == 0:
            # Prepare image observation. Currently this only supports simplerenv and should put this in the environment wrapper. 
            rgb_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].permute(0, 3, 1, 2)   # [N,C,H,W]
            # Convert to float and normalize to [0, 1] if needed
            if rgb_image.dtype == torch.uint8:
                rgb_image = rgb_image.float() / 255.0
            # Transfer image to model device if needed. but this step is slow though. maybe directly create one?
            if sim_device != model_device:
                rgb_image = rgb_image.to(model_device)
            img_list[camera_id]=rgb_image
        else:
            raise NotImplementedError(f"Currently we only support third-person view camera.")
    return img_list



