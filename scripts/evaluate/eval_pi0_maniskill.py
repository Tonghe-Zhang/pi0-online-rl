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

Config Override Feature:
You can now override num_steps, n_action_steps and act_steps from prompt for the model's config.json file
by adding config_overrides to your yaml configuration file:

model:
  model_overrides:
    num_steps: 10        # Override num_steps (default: 5)
    n_action_steps: 5    # Override n_action_steps (default: 50)
    act_steps: 5         # Override act_steps (default: 50)

Set values to null to use original config values.
See config/override_example.yaml for a complete example.

Evaluation script for pi0 model on ManiSkill3 environments.
By default, this script evaluates the pi0 model on the PutOnPlateInScene25Single-v1 task,
records videos in 10 environments, and logs the text prompts sent to the model.

Since we are doing evaluation, we focus on the episodic success rate, instead of the per-step reward which could be always zero according to the simulator. 

This script is designed to be efficient and save space when the environment number is large. 
It saves all environments' data in one file, creates one video for all environments,
and records all robot actions taken during the evaluation for analysis.

Data recorded includes:
- RGB observations (tiled videos)
- Robot actions (with statistics)
- Rewards and episode info
- Success rates per environment

Videos created:
- Regular tiled video: All environments in one video
- Success-filtered tiled video: All environments with green filter on successful episodes
- Success-only tiled video: Only successful environments (if any) 
"""

# DEBUG: Print CUDA_VISIBLE_DEVICES before any imports
import os
print(f"=== CUDA DEBUG INFO (BEFORE IMPORTS) ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
# When you use a HuggingFace tokenizer before multiprocessing, you may meet this warning:
# `The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...`
# This is because HuggingFace tokenizers use internal parallelism, 
# and when you run multi-GPU training with torchrun, 
# it forks processes after the tokenizer has already been loaded, which is unncessary and may cause deadlocks. 
# Consequently, we disable this parallelism by setting the environment variable TOKENIZERS_PARALLELISM to false
# when training on multiple GPUs. see https://github.com/huggingface/transformers/issues/5486 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# DEBUG: Check environment before torch import
print(f"Environment variables:")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"  TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM', 'Not set')}")

# import torch. This must be done after setting visible device variables. 
import torch
# DEBUG: Check CUDA after torch import
print(f"=== CUDA DEBUG INFO (AFTER TORCH IMPORT) ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
    
    # Try to get device properties to see physical GPU mapping
    try:
        import subprocess
        result = subprocess.run(['nvidia-ml-py3', '--list-gpus'], capture_output=True, text=True)
        print(f"Available physical GPUs: {result.stdout}")
    except:
        print("Could not get physical GPU info")


from colorama import init, Fore
import logging 
logger_custom = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import signal
import atexit
from pathlib import Path
from collections import defaultdict
from utils.reproduce import set_seed_everywhere
import hydra
from omegaconf import DictConfig
from mani_skill.envs.sapien_env import BaseEnv
from tqdm import tqdm as tqdm
from torch import Tensor 
# Add the parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c
# Import Customized helper files
from scripts.evaluate.eval_helpers import *
from scripts.env.env_helpers import setup_maniskill_env
from scripts.env.env_helpers import fetch_rgb_from_obs
from utils.custom_memory_manager import cleanup_cuda_memory, signal_handler
from utils.custom_dirs import PI_R_ROOT_DIR
from utils.clear_pycache import clean_pycache
from utils.custom_timer import current_time
# Register color font
init()
# Register cleanup functions
atexit.register(cleanup_cuda_memory)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
from lerobot.common.policies.pretrained import PreTrainedPolicy 


class EvalAgent:
    def __init__(self, cfg_eval):
        self.cfg_eval=cfg_eval
        # caching omega config
        # env
        self.env_name=cfg_eval.env.id
        self.num_envs=cfg_eval.env.num_envs
        self.n_steps_episode=cfg_eval.env.max_episode_len   # this one is passed to the environment constructor to define max_episode_steps. when elapsed_steps is larger than this value, environment returns `truncated`. 
        # sim
        self.sim_config=cfg_eval.env.sim_config
        self.sensor_config=cfg_eval.env.sensor_configs
        self.control_mode=cfg_eval.env.control_mode
        self.obs_mode=cfg_eval.env.obs_mode
        self.base_seed=cfg_eval.eval.seed
        # io paths
        self.dataset_stats_path=cfg_eval.dataset_stats
        self.model_path=cfg_eval.model.path
        self.output_dir=os.path.join(cfg_eval.output.dir, current_time())
        # create output directories. 
        self.video_dir, self.data_dir, self.log_dir=make_logging_dirs(self.output_dir)
        # log and visualize
        self.save_data=cfg_eval.output.save_data
        self.save_videos=cfg_eval.output.save_videos
        # determine device ids.
        sim_backend, sim_device, sim_device_id, model_device=set_model_sim_devices(self.cfg_eval)
        self.sim_backend=sim_backend
        self.sim_device=sim_device
        self.sim_device_id=sim_device_id
        self.model_device=model_device
        # create environment
        self.env: BaseEnv=setup_maniskill_env(env_id=self.env_name, 
                                              num_envs=self.num_envs, 
                                              max_episode_len=self.n_steps_episode, 
                                              sim_backend=self.sim_backend, 
                                              sim_device=self.sim_device, 
                                              sim_device_id=self.sim_device_id, 
                                              sim_config=self.sim_config, 
                                              sensor_config=self.sensor_config, 
                                              obs_mode=self.obs_mode, 
                                              control_mode=self.control_mode,
                                              episode_mode='eval')
        self.env_unwrapp: BaseEnv=self.env.unwrapped # type: ignore env.reward_mode==None, but it provides episodic success information in info['success']
        logger_custom.info(f"Environment setup complete. Initialized {self.env_name} with {self.num_envs} environments on {self.sim_device}.")

    def test(self, model: PreTrainedPolicy, verbose=True):
        """
        Test the policy in ManiSkill3 environment with 
        1. parallel evaluation with no reset in each episode. 
        2. one env, one episode, one rollout. 
        3. input dictionary (currently hard-coded but can be customized)
            batch = {
                        "observation.images.top": rgb_image,  # Tensor[B, C, H, W], this is transposed from simulator output [B,H,W,C], because PaliGemma image encoder receives images like [B,C,H,W], see code PATH_TO_YOUR_CONDA/envs/pi_r/lib/python3.10/site-packages/transformers/models/paligemma/modeling_paligemma.py function `get_image_features`
                        "observation.state": proprioception,  # Tensor[B, state_dim]
                        "task": instructions,                 # List of string of length B
                    }  
            the way we extract rgb data: obs["sensor_data"]["3rd_view_camera"]["rgb"]
        4. single-camera for now. 
        5. no reward in episodes, but records success rates for each episode. 
        """
        eval_prompt(self.cfg_eval)
        
        verbose_each_step=self.cfg_eval.verbose_each_step  
        
        clean_pycache(PI_R_ROOT_DIR)
        
        cleanup_cuda_memory()
        
        set_seed_everywhere(seed=self.base_seed)
        
        # save evaluation configuration. 
        save_config(self.cfg_eval, self.log_dir, 'cfg_eval')
        # save model configuration and architecture
        save_config(model.config, self.log_dir, 'cfg_model')
        save_model_architecture(model, self.log_dir)
        
        # Analyze the model to be evaluated
        num_denoising_steps=model.config.num_steps # type: ignore
        act_steps=model.config.act_steps   # action chunk size, open-loop control within it. # type: ignore
        # action_dim=self.env_unwrapp.single_action_space.shape[0]
        # analyze_space(env_unwrapp, env_unwrapp.single_action_space)
        
        # Calculate the chunk size / episode length. 
        chunk_episode_ratio = float(act_steps / self.n_steps_episode)
        if chunk_episode_ratio > 0.10:
            logger_custom.warning(
            f"{Fore.RED}act_steps={act_steps}, self.n_steps_episode={self.n_steps_episode}, action chunk size takes up {chunk_episode_ratio*100}% of the entire episode, this will make the policy not responsive enough! Consider adjust or overload your config file in {self.model_path}.{Fore.RESET}")
        else:
            if verbose:
                print(f"chunk_episode_ratio={chunk_episode_ratio}: act_steps={act_steps}, self.n_steps_episode={self.n_steps_episode}. ")
        
        # Initialize logging
        eval_metrics = defaultdict(list)
        logger_custom.info(f"Evaluating {model.__class__.__name__} model in environment {self.env_name} at {num_denoising_steps} step(s).")
        
        # prepare episode statistics billboards for each environment
        firsts_trajs = torch.zeros((self.n_steps_episode + 1, self.num_envs), device=self.sim_device)
        reward_trajs = torch.zeros((self.n_steps_episode, self.num_envs), device=self.sim_device)
        
        # reset environment to start this episode
        obs, step_info=reset_env_model(env=self.env, model=model, base_seed=self.base_seed, reset_model=True) # for pi-zero reset() only clears the action queue and will not affect training after testing. 
        obs: dict
        firsts_trajs[0] = 1
        reward=torch.zeros(self.num_envs, device=self.sim_device)
        # Get language instructions
        instructions: List[str] = self.env_unwrapp.get_language_instruction() # a list of strings with len=num_envs # type: ignore
        # Create batch episode data storage
        episode_data = create_batch_episode_data(self.num_envs, instructions)
        # Initial step: no action taken yet, pass None
        episode_data.add_step_data_in_batch(obs["sensor_data"]["3rd_view_camera"]["rgb"], reward, step_info, actions_batch=None)  # type: ignore
        
        for step in tqdm(range(self.n_steps_episode), dynamic_ncols=True):
            with torch.no_grad():
                # Prepare inputs
                rgb_image:Tensor=fetch_rgb_from_obs(obs, self.sim_device, self.model_device)[0]        # type: ignore [B, C, H, W], where B here is the self.num_envs
                proprioception: Tensor =self.env_unwrapp.agent.robot.get_qpos().to(self.model_device)  # qpos
                batch = {
                    "observation.images.top": rgb_image,  # Tensor[B, C, H, W], this is transposed from simulator output [B,H,W,C], because PaliGemma image encoder receives images like [B,C,H,W], see code PATH_TO_YOUR_CONDA/envs/pi_r/lib/python3.10/site-packages/transformers/models/paligemma/modeling_paligemma.py function `get_image_features`
                    "observation.state": proprioception,  # Tensor[B, state_dim]
                    "task": instructions,                 # List of string of length B
                }
                if verbose:
                    print(f"Inputs to pi-zero: rgb_image: {rgb_image.shape}, proprioception_state={proprioception.shape}, task instructions: {instructions}")
                
                # Get action from pi-zero, actually poping out the oldest action in the action chunk queue (currently open-loop control within the chunk). 
                action = model.select_action(batch)
            
            # Step environment
            action = action.to(self.sim_device) if self.sim_device != self.model_device else action      
            next_obs, reward, terminated, truncated, step_info = ret_batch = self.env.step(action) # parallel sim      
            
            # Log this transition      
            if verbose_each_step:
                log_step(action, ret_batch)
            # Record step data in batch
            episode_data.add_step_data_in_batch(obs["sensor_data"]["3rd_view_camera"]["rgb"], reward, step_info, actions_batch=action) # type: ignore
            
            # Proceeed to next step
            reward_trajs[step] = reward
            firsts_trajs[step + 1] = torch.logical_or(terminated,truncated)
            obs = next_obs # type: ignore
            print_batch_step_info(step, episode_data)
        
        # Add the final image observation (step self.n_steps_episode+1) and summarize the whole episode's success rate
        episode_data.add_final_observation_in_batch(obs["sensor_data"]["3rd_view_camera"]["rgb"]) # type: ignore
        episode_data.finalize_success_status()
        
        # Save episode data for all environments together
        if self.save_data:
            save_batch_episode_data(episode_data, self.data_dir)
        
        # Create videos efficiently
        if self.save_videos:
            create_batch_videos(episode_data, self.video_dir, fps=10)
        
        # Update metrics
        for env_idx in range(self.num_envs):
            eval_metrics["success"].append(episode_data.success[env_idx])
            if episode_data.infos[env_idx]: # Add other metrics if available, for example if the object is grasped. 
                final_info = episode_data.infos[env_idx][-1]
                for key in ["is_src_obj_grasped", "consecutive_grasp"]:
                    if key in final_info:
                        eval_metrics[key].append(final_info[key])
        # clean memory and handling errors
        cleanup_cuda_memory()
        # Summmarize and log episode statistics. 
        success_rate=log(self.env_name, self.model_path, self.num_envs, self.base_seed, eval_metrics, self.log_dir)
        return success_rate

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg_eval: DictConfig):
    """Evaluation wrapper"""
    # create evaluation environment
    eval_agent=EvalAgent(cfg_eval)
    
    # Load model
    model:PI0Policy
    model_overrides=cfg_eval.model.get('model_overrides', {})
    model=load_pi0_model(eval_agent.model_path, eval_agent.model_device, eval_agent.dataset_stats_path, model_overrides)
    
    # start evaluation
    try:
        eval_agent.test(model, verbose=cfg_eval.verbose)
        # Showcase the directories where we saved the eval data. 
        print(f"\nResults saved to: {eval_agent.output_dir}")
        print(f"Videos saved to: {eval_agent.video_dir}")
        print(f"Data saved to: {eval_agent.data_dir}")
        print(f"Logs saved to: {eval_agent.log_dir}")
        print("\nEvaluation completed!")
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user. Cleaning up...")
        cleanup_cuda_memory()
        return
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        cleanup_cuda_memory()
        raise
    finally:
        if eval_agent.env:
            eval_agent.env.close()
        cleanup_cuda_memory()
    
if __name__ == "__main__":
    main()