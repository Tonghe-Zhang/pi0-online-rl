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


from typing import Dict, Any, List
from omegaconf import DictConfig
import torch
from policy.p0_ppo import PI0PolicyPPO
from lerobot.common.policies.pi0.modeling_pi0_rl import PI0Config
import logging
logger = logging.getLogger(__name__)
from termcolor import colored
def create_model(cfg: DictConfig, dataset_stats: Dict[str, Any], device: torch.device) -> tuple[PI0PolicyPPO, PI0Config]:
    """Create and initialize the Pi0 model from the folder that stores the checkpoint, it can either point to the main model (SFT/RLFT) or the ema model (optionally from SFT.)
    parent_checkpoint_dir
        ├── model/-----------------------------------------------------------------------------------------> cfg.model.path 
        │   ├── config.json  # policy config (fixed to remove 'type' field and normalize device)
        │   ├── model.safetensors  # policy weights
        ├── ema_model/ #(optional)-------------------------------------------------------------------------> or cfg.model.path 
        │   ├── config.json  # policy config (fixed to remove 'type' field and normalize device)
        │   ├── model.safetensors  # policy weights
        |__ training_state/
            |...
    """
    
    model_path = cfg.model.path 
    logger.info(f"Loading model from {model_path}")
    
    # Load original model config
    model_config: PI0Config = PI0Config.from_pretrained(model_path)
    
    # Overridge model_config with command line inputs in training config file
    model_config_overrides = cfg.model.get('model_config_overrides', {})
    model_config_overrides = {k: v for k, v in model_config_overrides.items() if v is not None} if model_config_overrides else None
    if model_config_overrides:
        logger.info("Applying model_config_overrides:")
        for key, value in model_config_overrides.items():
            if value is not None:
                original_value = getattr(model_config, key, None)
                logger.info(f"  Overriding {key}: {original_value} -> {value}")
                setattr(model_config, key, value)
    else:
        logger.info(colored("Warning: No model_config_overrides provided.", "red", "on_red"))
    
    # Set training configuration to model config. 
    model_config.freeze_vision_encoder = cfg.model.freeze_vision_encoder
    model_config.train_expert_only = cfg.model.train_expert_only
    model_config.device = str(device)
    
    logger.info(f"Appending RLFT config to pre-trained model config...")
    # Append the model_config with RLFT specific parameters, to make it a config for PI0PolicyRL
    for k, v in cfg.rlft_config.items():
        logger.info(f"""From rlft_config["{k}"]={v}""")
        setattr(model_config, k, v)
        logger.info(f"After setting: model_config.{k}={getattr(model_config, k)}")
    
    logger.info(f"\nFinal configuration before loading model: model_config={model_config}\n")
    
    
    # Create model from the overridden model config. 
    model = PI0PolicyPPO.from_pretrained(
        pretrained_name_or_path=model_path,
        config=model_config,
        dataset_stats=dataset_stats
    )
    
    model.to(device)
    logger.info(f"Successfully loaded Pi-zero model on {device}")
    
    return model, model_config

import os
import json
from pathlib import Path
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from lerobot.common.policies.pretrained import PreTrainedPolicy
from scripts.sft.sft_helpers import save_model_core
from lerobot.common.optim.optimizers import save_optimizer_state
from lerobot.common.optim.schedulers import save_scheduler_state
def save_rlft_training_state(
    checkpoint_dir: Path,
    itr:int,
    cnt_train_step: int,
    actor_optimizer: Optimizer | None = None,
    actor_scheduler: LRScheduler | None = None,
    critic_optimizer: Optimizer | None = None,
    critic_scheduler: LRScheduler | None = None) -> None:
    """
    Saves the training step, optimizer state, scheduler state, and rng state.

    Args:
        save_dir (Path): The directory to save artifacts to.
        itr (int): RLFT training iteration. 
        cnt_train_step (int):  RLFT training sample complexity, in terms of transition tuples. 
        actor_optimizer: Optimizer | None = None,
        actor_scheduler: LRScheduler | None = None,
        critic_optimizer: Optimizer | None = None,
        critic_scheduler: LRScheduler | None = None
    
    Result:
    
  checkpoint_dir/
        |__ training_state/
                |__metadata.json
                |__actor/
                |   ├── optimizer_param_groups.json  #  optimizer param groups
                |   |── optimizer_state.safetensors  # optimizer state
                |   ├── scheduler_state.json  # scheduler state
                |___critic/
                    ├── optimizer_param_groups.json  #  optimizer param groups
                    |── optimizer_state.safetensors  # optimizer state
                    ├── scheduler_state.json  # scheduler state
    """
    save_dir = checkpoint_dir/'training_state'
    os.makedirs(save_dir, exist_ok=True)
    
    actor_save_dir = save_dir/'actor'
    os.makedirs(actor_save_dir, exist_ok=True)
    
    critic_save_dir = save_dir/ 'critic'
    os.makedirs(critic_save_dir, exist_ok=True)
    
    # save itr and cnt_train_step
    metadata = {
        "itr": itr,
        "cnt_train_step": cnt_train_step
    }
    with open(os.path.join(save_dir, "rlft_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    # save actor/critic optimizers and schedulers
    if actor_optimizer is not None:
        save_optimizer_state(actor_optimizer, actor_save_dir)
    else:
        raise ValueError(f"Saving actor_optimizer={actor_optimizer}. Check your data saving logic. ")
    if actor_scheduler is not None:
        save_scheduler_state(actor_scheduler, actor_save_dir)
    else:
        raise ValueError(f"Saving actor_scheduler={actor_scheduler}. Check your data saving logic. ")
    if critic_optimizer is not None:
        save_optimizer_state(critic_optimizer, critic_save_dir)
    else:
        raise ValueError(f"Saving critic_optimizer={critic_optimizer}. Check your data saving logic. ")
    if critic_scheduler is not None:
        save_scheduler_state(critic_scheduler, critic_save_dir)
    else:
        raise ValueError(f"Saving critic_scheduler={critic_scheduler}. Check your data saving logic. ")

def save_rlft_model_best(checkpoint_dir: Path,
                        itr: int,
                        cnt_train_step: int,
                        is_current_best: bool, 
                        model:PreTrainedPolicy,
                        actor_optimizer: Optimizer,
                        critic_optimizer: Optimizer,
                        actor_scheduler: LRScheduler | None = None,
                        critic_scheduler: LRScheduler | None = None):
    """This function creates the following directory structure:
    
    checkpoint_dir/
        |__ last/
        |    ├── model/
        |    │   ├── config.json  # policy config (fixed to remove 'type' field and normalize device)
        |    │   ├── model.safetensors  # policy weights
        |    |__ training_state/
        |           |__metadata.json
        |           |__actor/
        |           |   ├── optimizer_param_groups.json  #  optimizer param groups
        |           |   |── optimizer_state.safetensors  # optimizer state
        |           |   ├── scheduler_state.json  # scheduler state
        |           |___critic/
        |               ├── optimizer_param_groups.json  #  optimizer param groups
        |               |── optimizer_state.safetensors  # optimizer state
        |               ├── scheduler_state.json  # scheduler state
        |__ best/
            ├── model/
            │   ├── config.json  # policy config (fixed to remove 'type' field and normalize device)
            │   ├── model.safetensors  # policy weights
            |__ training_state/
                    |__metadata.json
                    |__actor/
                    |   ├── optimizer_param_groups.json  #  optimizer param groups
                    |   |── optimizer_state.safetensors  # optimizer state
                    |   ├── scheduler_state.json  # scheduler state
                    |___critic/
                        ├── optimizer_param_groups.json  #  optimizer param groups
                        |── optimizer_state.safetensors  # optimizer state
                        ├── scheduler_state.json  # scheduler state

    Args:
        checkpoint_dir (Path): The root directory to save the checkpoints
        step (int): The training step at that checkpoint.
        is_current_best (bool): 
        policy (PreTrainedPolicy): The policy to save.
        optimizer (Optimizer | None, optional): The optimizer to save the state from. Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler to save the state from. Defaults to None.
        
    """
    model_to_save=model.module if isinstance(model, DDP) else model
    
    # save the last checkpoint
    save_rlft_training_state(checkpoint_dir/'last', itr, cnt_train_step, actor_optimizer, actor_scheduler, critic_optimizer, critic_scheduler)
    model_path=checkpoint_dir/'last'/'model'
    model_path.mkdir(parents=True, exist_ok=True)
    save_model_core(model_to_save, save_directory=model_path) # Use our custom save function that fixes the 'type' and device issues
    
    # save the best checkpoint
    if is_current_best:
        best_model_path=checkpoint_dir/'best'/'model'
        best_model_path.mkdir(parents=True, exist_ok=True)
        save_rlft_training_state(checkpoint_dir/'best', itr, cnt_train_step, actor_optimizer, actor_scheduler, critic_optimizer, critic_scheduler)
        save_model_core(model_to_save, save_directory=best_model_path)

def log_step_simple(step:int, ret_batch, done_venv):
    next_obs, reward, terminated, truncated, step_info=ret_batch
    logger.info(f"\nSTEP={step}\nterminated={terminated.shape}, {terminated.device}, {list(terminated.cpu().numpy())}\
        \ntruncated={truncated.shape}, {truncated.device},{list(truncated.cpu().numpy())}\
            \ndone_venv={done_venv.shape}, {done_venv.device}, {list(done_venv.cpu().numpy())}\
                \nstep_info-success={step_info['success'].device}, {list(step_info['success'].cpu().numpy())}\
                    \nstep_info-is_src_obj_grasped={list(step_info['is_src_obj_grasped'].cpu().numpy())}\
                        \nreward={list(reward.cpu().numpy())}")



import numpy as np
from scripts.evaluate.eval_helpers import tile_images, images_to_video
class AsyncVideoRecorder:
    def __init__(self, num_envs: int, instructions: List[str]):
        self.num_envs = num_envs
        self.instructions = instructions
        self.all_frames = []  # List of [B, H, W, C] numpy arrays (tiled later)
        self.firsts_trajs = np.ones((1, num_envs), dtype=bool)  # Start with initial 'reset'
        self.success_per_episode = []  # List of (env_idx, start_step, end_step, episode_success)
        self.has_succeeded = [False] * num_envs  # Running flag per env
        self.global_step = 0
        self.total_episodes = 0
        self.min_episode_length = 2  # Align with buffer's skip of length <=1

    def align_stats_with_buffer(self, eval_buffer):
        self.success_rate = eval_buffer.success_rate
        self.n_successful_episodes = eval_buffer.n_successful_episodes
        self.total_episodes = eval_buffer.n_episode_finished
        logger.info(f"Aligned video stats with eval_buffer: success_rate={self.success_rate}, n_successful={self.n_successful_episodes}, total_episodes={self.total_episodes}")

    def add_frame(self, rgb_batch: torch.Tensor, done_venv: torch.Tensor, info_venv: dict, is_step: bool = True):
        # Store frame always
        rgb_array = rgb_batch.cpu().numpy()
        if rgb_array.max() <= 1.0:
            rgb_array = (rgb_array * 255).astype(np.uint8)
        self.all_frames.append(rgb_array)

        # Get per-env info from {'episode_infos': [dict_env0, dict_env1, ...]}
        episode_infos = info_venv.get('episode_infos', [{} for _ in range(self.num_envs)])

        # Track resets and update success
        current_firsts = np.zeros(self.num_envs, dtype=bool)
        for env_idx in range(self.num_envs):
            # Update running success flag from current step
            success_this_step = episode_infos[env_idx].get('success', False)
            if success_this_step:
                self.has_succeeded[env_idx] = True

            if done_venv[env_idx].item():
                current_firsts[env_idx] = True
                if self.global_step > 0:
                    prev_start = np.where(self.firsts_trajs[:, env_idx])[0][-1]
                    length = self.global_step - prev_start + 1
                    episode_success = self.has_succeeded[env_idx]
                    if length >= self.min_episode_length:
                        self.success_per_episode.append((env_idx, prev_start, self.global_step, episode_success))
                        self.total_episodes += 1
                    self.has_succeeded[env_idx] = False  # Reset for new episode

        # Only append to firsts_trajs if this is an actual step (not initial or final)
        if is_step:
            self.firsts_trajs = np.vstack([self.firsts_trajs, current_firsts])
        self.global_step += 1

    def finalize(self):
        # Finalize open episodes
        for env_idx in range(self.num_envs):
            if not self.firsts_trajs[-1, env_idx]:
                start = np.where(self.firsts_trajs[:, env_idx])[0][-1]
                length = self.global_step - start + 1
                success = self.has_succeeded[env_idx]
                if length >= self.min_episode_length:
                    self.success_per_episode.append((env_idx, start, self.global_step, success))
                    self.total_episodes += 1

    def align_with_buffer(self, buffer):
        # Clear existing tracking
        self.success_per_episode = []
        self.total_episodes = 0
        self.n_successful_episodes = 0

        # Use buffer's firsts_trajs and reward_trajs directly (convert to np)
        firsts_np = buffer.firsts_trajs.cpu().numpy()  # [n_steps+1, n_envs]
        reward_np = buffer.reward_trajs.cpu().numpy()  # [n_steps, n_envs]

        # Mirror buffer's episode detection logic exactly
        episodes_start_end = []
        for env_ind in range(self.num_envs):
            env_steps = np.where(firsts_np[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1] - 1  # Inclusive, matches buffer
                if (end - start + 1) > 1:  # Matches buffer's >1 filter
                    episodes_start_end.append((env_ind, start, end))

        # For each detected episode, compute success using buffer's reward threshold
        for env_ind, start, end in episodes_start_end:
            ep_rewards = reward_np[start:end+1, env_ind]  # Exact slice
            max_chunk_reward = np.max(ep_rewards)
            ep_success = max_chunk_reward >= buffer.success_rew_threshold_in_chunk
            self.success_per_episode.append((env_ind, start, end, ep_success))  # Note: end is inclusive
            self.total_episodes += 1
            if ep_success:
                self.n_successful_episodes += 1

        # Compute rate naturally (will match buffer)
        self.success_rate = self.n_successful_episodes / self.total_episodes if self.total_episodes > 0 else 0.0

        logger.info(f"Video episode detection aligned with buffer: {self.total_episodes} episodes, {self.n_successful_episodes} successes, rate={self.success_rate:.4f}")

    def _generate_masked_frames(self, apply_mask: bool = True):
        """
        Optionally apply mask to the frames if it belongs to a successful episode. 
        """
        frames = []
        for step in range(self.global_step):
            frame = self.all_frames[step]
            masks = [False] * self.num_envs
            for env_idx in range(self.num_envs):
                for e_env, e_start, e_end, e_success in self.success_per_episode:
                    if e_env == env_idx and e_start <= step < e_end and e_success:
                        masks[env_idx] = True
                        break
            if apply_mask:
                tiled = tile_images(frame, success_flags=masks)
            else:
                tiled = tile_images(frame)
            frames.append(tiled)
        return frames

    def _generate_success_only_frames(self):
        """
        The successful episodes of each environment are padded to the same length with the last frame, and then tiled. 
        """
        successful_episodes = [e for e in self.success_per_episode if e[3]]
        if not successful_episodes:
            return []

        # Find max length (in buffer steps, then convert to frames)
        max_len = max(end - start + 1 for _, start, end, _ in successful_episodes)

        # Pad each successful episode's frames (adjust for video's extra initial frame)
        padded_ep_frames = []
        for env_idx, start, end, _ in successful_episodes:
            # Buffer start/end are in step indices (0 to n_steps-1)
            # Video frames: [0]=initial, [1]=after step0, ..., [n_steps]=after last step, [n_steps+1]=final
            # So, episode from buffer start to end corresponds to video frames (start+1) to (end+1)
            ep_frames = [self.all_frames[s + 1][env_idx] for s in range(start, end + 1)]
            pad_len = max_len - len(ep_frames)
            if pad_len > 0:
                last_frame = ep_frames[-1]
                ep_frames.extend([last_frame] * pad_len)
            padded_ep_frames.append(ep_frames)

        # Generate tiled frames for each timestep
        tiled_frames = []
        for t in range(max_len):
            step_frames = [ep[t] for ep in padded_ep_frames]
            step_array = np.stack(step_frames)
            tiled = tile_images(step_array)
            tiled_frames.append(tiled)
        return tiled_frames

    def save_video(self, video_dir: Path, fps: int = 10, buffer=None):
        self.finalize()
        if buffer:
            self.align_with_buffer(buffer)
        success_rate = self.success_rate if hasattr(self, 'success_rate') else np.mean([episode_success_flag for _, _, _, episode_success_flag in self.success_per_episode]) if self.success_per_episode else 0.0
        filename_base = f"async_success_rate_{success_rate:.2f}"

        # 1. All envs no mask
        no_mask_frames = self._generate_masked_frames(apply_mask=False)
        no_mask_frames_path = f"{filename_base}_all_no_mask"
        images_to_video(no_mask_frames, str(video_dir), no_mask_frames_path, fps=fps, verbose=False)
        logger.info(f"  ✓ All envs no mask video saved to {no_mask_frames_path}")
        
        # 2. All envs with masks
        masked_frames = self._generate_masked_frames(apply_mask=True)
        masked_frames_path = f"{filename_base}_all_masked"
        images_to_video(masked_frames, str(video_dir), masked_frames_path, fps=fps, verbose=False)
        logger.info(f"  ✓ All envs masked video saved to {masked_frames_path}")

        # 3. Successful episodes only (no masks)
        success_only_frames = self._generate_success_only_frames()
        if success_only_frames:
            success_only_frames_path = f"{filename_base}_success_only"
            images_to_video(success_only_frames, str(video_dir), success_only_frames_path, fps=fps, verbose=False)
            logger.info(f"  ✓ Successful episodes only video saved to {success_only_frames_path}")
        else:
            logger.info(f"  ⚠ No successful episodes - skipping success-only video.")
        logger.info(f"Videos captured with success_rate={success_rate} and successful episodes number {self.n_successful_episodes if hasattr(self, 'n_successful_episodes') else 0} and total episode number {self.total_episodes}")
            

# not used anywhere now. 
def minibatch_generator_denoising(self):
    self.approx_kl = 0.0
    
    obs, chains, returns, oldvalues, advantages, oldlogprobs =  self.buffer.make_dataset()
    # Explained variation of future rewards using value function
    self.explained_var = self.buffer.get_explained_var(oldvalues, returns)
    # define Q values as the old Q values to align with the definition in diffusion ppo. you can change those back to new Q values 
    self.Q_values = oldvalues.mean().item()
    
    self.total_steps = self.n_steps_rollout * self.n_envs * self.n_denoising_steps  
    for update_epoch in range(self.update_epochs):
        self.kl_change_too_much = False
        indices = torch.randperm(self.total_steps, device=self.buffer_device)
        for batch_id, start in enumerate(range(0, self.total_steps, self.batch_size)):
            end = start + self.batch_size
            inds_b = indices[start:end]                            # b is for batchsize
            
            # split the indices into batch_inds_b and denoising_inds_b, 
            # where batch_inds_b corresponds to the id in self.n_steps_rollout * self.n_envs
            # and denoising_inds_b corresponds to the id in self.n_denoising_steps
            batch_inds_b, denoising_inds_b = torch.unravel_index(
                inds_b,
                (self.n_steps_rollout * self.n_envs, self.n_denoising_steps),
            )
            minibatch = (
                {k: v[batch_inds_b].to(device=self.model_device, dtype=self.model_dtype) if isinstance(v, torch.Tensor)       # visuomotor (tensor) obs
                    else [v[i] for i in batch_inds_b]                                                                      # language (lists of strings) obs
                    for k, v in obs.items()},                         
                chains[batch_inds_b, denoising_inds_b].to(device=self.model_device, dtype=self.model_dtype),            # chains_prev_b    self.batch_size x self.horizon_steps x self.act_dim 
                chains[batch_inds_b, denoising_inds_b + 1].to(device=self.model_device, dtype=self.model_dtype),        # chains_next_b    self.batch_size x self.horizon_steps x self.act_dim 
                denoising_inds_b.to(device=self.model_device, dtype=torch.long),                                        # denoising_inds_b [B,] long tensor for indexing. 
                returns[batch_inds_b].to(device=self.model_device, dtype=self.model_dtype),                             # returns_b        [B,] 
                oldvalues[batch_inds_b].to(device=self.model_device, dtype=self.model_dtype),                           # values_b         [B,] 
                advantages[batch_inds_b].to(device=self.model_device, dtype=self.model_dtype),                          # advantages_b     [B,]  there are many duplicated entries.
                oldlogprobs[batch_inds_b, denoising_inds_b].to(device=self.model_device, dtype=self.model_dtype)        # logprobs_b       self.batch_size x self.denoising_steps x self.horizon_steps x self.act_dim 
            )
            
            if self.lr_schedule=='fixed' and self.target_kl and self.approx_kl > self.target_kl: # we can also use adaptive KL instead of early stopping.
                self.kl_change_too_much = True
                logger.warning(f"KL change too much, approx_kl ={self.approx_kl} > {self.target_kl} = target_kl, stop optimization.")
                break
            
            yield update_epoch, batch_id, minibatch    
    
