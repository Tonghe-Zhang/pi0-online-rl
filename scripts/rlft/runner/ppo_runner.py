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
Places need to be changed:
0. The eval script is a bit problematic... maybe we should use sync eval for fair comparison. align with the eval script in sft. 
1. tune hyperparams based on chenkang's.
2. implement and ablate torque-level domain randomization.
"""
# Standard libraries
import os
import json
import numpy as np
import torch
from torch import Tensor
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
# Logging and visualize
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm as tqdm
from matplotlib import pyplot as plt
import pickle
import wandb
from typing import List
from pathlib import Path
from termcolor import colored
import matplotlib.pyplot as plt
# Register cleanup functions
import signal
import atexit
from utils.custom_memory_manager import cleanup_cuda_memory, signal_handler, print_memory_usage
atexit.register(cleanup_cuda_memory)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
# Custom libs
from buffer import PPOEvalBufferGPU, PPOFlowImgBufferGPU
from utils.custom_scheduler import get_scheduler, CustomScheduler
from utils.custom_timer import Timer
from utils.custom_logging import create_bordered_text
from utils.reward_scaling import RunningRewardScaler
from utils.custom_scheduler import CustomScheduler, CosineAnnealingWarmupRestarts
from scripts.evaluate.eval_helpers import set_model_sim_devices, save_model_architecture, BatchEpisodeData, create_batch_videos
from scripts.env.env_helpers import setup_maniskill_env, fetch_rgb_from_obs
from scripts.rlft.runner.rlft_helpers import create_model, save_rlft_model_best, log_step_simple, AsyncVideoRecorder
from scripts.sft.sft_helpers import save_model_config
# Model
from runner.base_runner import BaseRunner
from rlft.policy.p0_ppo import PI0PolicyPPO
from scripts.env.domain_rand import DomainRandomization
from lerobot.common.optim.optimizers import load_optimizer_state
from lerobot.common.optim.schedulers import load_scheduler_state

class PPORunner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.policy: PI0PolicyPPO
        # PPO parameters
        # Batch size for PPO gradient update
        self.batch_size = cfg.train.batch_size
        # Disconut factor
        self.gamma = cfg.train.gamma
        # Generalized advantage estimation
        self.gae_lambda: float = cfg.train.get("gae_lambda", 0.95)
        # If specified, stop gradient update once KL difference reaches it
        self.target_kl: float = cfg.train.get("target_kl", 0.1)
        # Clip the norms of actor and critic below `self.max_grad_norm`
        self.max_grad_norm = cfg.train.get("max_grad_norm", None)
        # Number of times the collected data is used in gradient update
        self.update_epochs: int = cfg.train.update_epochs
        # Gradient accumulation to deal with large GPU RAM usage
        self.grad_accumulate = cfg.train.grad_accumulate
        # Entropy loss coefficient
        self.ent_coef = cfg.train.get("ent_coef", 0.01)
        # Value loss coefficient
        self.vf_coef = cfg.train.get("vf_coef", 0.5)
        # Advantage estimate method
        self.adv_method=cfg.train.get("adv_method", "GAE")
        # The logprob ratio should be strictly 1.00 when batch=0 and epoch=0. 
        # However, if we add randomization in the buffer that could be different from batch to batch, then we can allow for certain degree of error. 
        self.initial_ratio_error_threshold = 1e-6 
        
        # Reward configuration
        # Whether to use running reward scaling
        self.reward_scale_running: bool = cfg.reward.reward_scale_running
        if self.reward_scale_running:
            self.running_reward_scaler = RunningRewardScaler(self.n_envs)
        # Scaling reward with constant
        self.reward_scale_const: float = cfg.reward.get("reward_scale_const", 1.0)
        # Whether to ignore the next-step value for truncated rollout steps.
        self.ignore_nextvalue_when_truncated: bool = cfg.reward.ignore_nextvalue_when_truncated
        # The lowest action chunk aggregated reward to indicate a success. 
        self.success_rew_threshold_in_chunk: float = cfg.reward.success_rew_threshold_in_chunk
        # Record best performance
        self.current_best_reward = np.float32('-inf')
        self.is_current_best = False 
        # Stop when performance is too bad. 
        self.use_early_stop = cfg.train.use_early_stop
        
        # Action configuration
        self.n_denoising_steps=self.policy.config.num_steps                                   # flow ODE integration steps
        self.n_action_steps=self.policy.config.n_action_steps                                 # action chunk size (model's output). open-loop control within it. 
        self.act_steps = self.cfg.rlft_config.act_steps                                       # action chunk size (actually deployed). By default. we use all the output action chunk.  
        assert self.act_steps <=self.n_action_steps, f"act_steps (deployed chunk size)={self.act_steps} must be less than or equal to n_action_steps (model's output chunk size)={self.n_action_steps}"
        assert self.act_steps == self.policy.config.act_steps, f"act_steps (deployed chunk size)={self.act_steps} must be equal to model.config.act_steps={self.policy.config.act_steps}"
        self.max_action_dim = self.policy.config.max_action_dim                               # the maximum action dimension that the model can handle, which is the dimension of the model's raw output. 
        self.effective_action_dim=self.env_unwrapp.single_action_space.shape[0]               # type: ignore the effective action dimension for THIS robot.
        
        # Observation configuration
        self.n_cond_step = self.policy.config.n_obs_steps                                     # currently we do not concate historical observations, whether it is the state or the images. 
        self.rgb_keys: List[str]= self.cfg.env.obs.rgb_keys                                   # a list of strings indicating the camera names. 
        self.proprioception_key: str = self.cfg.env.obs.proprioception_key
        self.language_key: str = self.cfg.env.obs.language_key
        self.visuomotor_metadata:dict = self.policy.config.input_features                     # only records visual and proprioception keys and shapes. 
        self.visuomotor_obs_dict={k: self.visuomotor_metadata[k].shape for k in self.visuomotor_metadata.keys()}  # {"observation.images.top": [3, 480, 640] , "observation.state": [8]}. This one should align with simulator output and pi0's config.json
        logger.info(f"visuomotor_obs_dict={self.visuomotor_obs_dict}")
        
        # (Optional) BC regularization
        self.use_bc_loss: bool = cfg.train.get("use_bc_loss", False)
        self.bc_loss_type: str = cfg.train.get("bc_loss_type", None) if self.use_bc_loss else None # type: ignore
        bc_loss_type_candidates=['W2', 'KL', None]
        if not self.bc_loss_type in bc_loss_type_candidates:
            raise ValueError(f"Unsupported bc_loss_type={self.bc_loss_type}. It must be in the following: {bc_loss_type_candidates}")
        self.bc_coeff: float = cfg.train.get("bc_loss_coeff", 0.0)
        
        # Memory management configuration
        self.memory_cleanup_freq_rollout: int = cfg.memory.memory_cleanup_freq_rollout  # Cleanup every N steps during rollout
        self.memory_cleanup_freq_grad: int = cfg.memory.memory_cleanup_freq_grad  # Cleanup every N gradient updates
        self.enable_aggressive_cleanup: bool = cfg.memory.enable_aggressive_cleanup  # Enable more frequent cleanup
        
        # Whether to clip the denoised actions to the range [model.config.denoise_action_min, model.config.denoise_action_max]
        self.clip_intermediate_actions = cfg.train.get("clip_intermediate_actions", False)
        if self.clip_intermediate_actions and self.policy.config.__dict__.get('denoise_action_min', None) is None:
            raise ValueError("clip_intermediate_actions is True, but model.config.denoise_action_min is None. Please set model.config.denoise_action_min to the minimum possible value of the action space, or use clip_intermediate_actions=False when there is no explicit physical constraint on the action space.")
        if self.clip_intermediate_actions and self.policy.config.__dict__.get('denoise_action_max', None) is None:
            raise ValueError("clip_intermediate_actions is True, but model.config.denoise_action_max is None. Please set model.config.denoise_action_max to the maximum possible value of the action space, or use clip_intermediate_actions=False when there is no explicit physical constraint on the action space.")
        
        # (For Diffusion/Flow models) Noise scheduler for SDE-based denoising. 
        self.build_noise_scheduler()
        
        # Create buffer
        self.create_buffer()
        
        self.verbose_each_step = cfg.logging.verbose_each_step
        self.verbose_sampling_progress=cfg.logging.verbose_sampling_progress
        self.verbose_buffer_filling=cfg.logging.verbose_buffer_filling  
    
    def prepare_video_recording(self):
        # Create video directory
        self.video_dir = Path(self.render_dir) / "training_videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        
        # Get language instructions from environment
        language_instructions = self.env_unwrapp.get_language_instruction() # type: ignore
        
        # Create async video recorder
        self.video_recorder = AsyncVideoRecorder(self.n_envs, language_instructions)
        
        # Record initial frame
        initial_rgb = self.obs_venv["sensor_data"]["3rd_view_camera"]["rgb"]  # type: ignore # [B, H, W, C]
        self.video_recorder.add_frame(initial_rgb, torch.zeros(self.n_envs, dtype=torch.bool, device=self.sim_device), {}, is_step=False) # type: ignore
        logger.info(f"Async video recording initialized for {self.n_envs} environments")
    
    def record_video_frame(self, info_venv: dict):
        current_rgb = self.obs_venv["sensor_data"]["3rd_view_camera"]["rgb"]  # type: ignore # [B, H, W, C]
        if not isinstance(current_rgb, Tensor):
            current_rgb = torch.tensor(current_rgb)
        
        # Convert info_venv to per-env dicts for success checking
        info_per_env = [{} for _ in range(self.n_envs)]
        for key, value in info_venv.items():
            if isinstance(value, (list, tuple, np.ndarray, Tensor)) and len(value) == self.n_envs:
                for i in range(self.n_envs):
                    info_per_env[i][key] = value[i].item() if hasattr(value[i], 'item') else value[i]
            else:
                for i in range(self.n_envs):
                    info_per_env[i][key] = value
        
        self.video_recorder.add_frame(current_rgb, self.done_venv, {'episode_infos': info_per_env}, is_step=True) # type: ignore
    
    def save_video(self, buffer=None):
        # Add final frame (visual only)
        final_rgb = self.obs_venv["sensor_data"]["3rd_view_camera"]["rgb"]  # type: ignore
        self.video_recorder.add_frame(final_rgb, torch.zeros(self.n_envs, dtype=torch.bool, device=self.sim_device), {}, is_step=False)  # type: ignore
        
        # Save with alignment if buffer provided
        self.video_recorder.save_video(self.video_dir, fps=10, buffer=buffer)  # type: ignore
        logger.info(f"Async video saved to: {self.video_dir}")
    
    def record_video(self):
        record_options =['first', 'last', 'periodic']
        if self.record_video_condition == 'first':
            return self.itr == 0
        elif self.record_video_condition == 'last':
            return self.itr == self.n_train_itr - 1
        elif self.record_video_condition == 'periodic':
            return self.itr % self.video_freq == 0
        else:
            raise ValueError(f"Invalid record_video_condition={self.record_video_condition}. It must be in the following: {record_options}")
    
    def run(self):
        self.prepare_run()
        self.buffer.reset() # as long as we put items at the right position in the buffer (determined by 'step'), the buffer automatically resets when new iteration begins (step =0). so we only need to reset in the beginning. This works only for PPO buffer, otherwise may need to reset when new iter begins.
        self.eval_buffer.reset()
        if self.resume:
            self.resume_training_state()  # Recover the optimizer, scheduler, itr number... This function is not in charge of loading models. 
        while self.itr < self.n_train_itr:
            # Monitor memory usage at the start of each iteration
            if self.enable_aggressive_cleanup:
                print_memory_usage()
            
            self.set_model_mode()
            self.reset_env_model()
            if self.record_video():
                self.prepare_video_recording()
            
            for step in tqdm(range(self.n_steps_rollout)) if self.verbose_sampling_progress else range(self.n_steps_rollout):
                if not self.verbose_sampling_progress and step % 50 == 0: logger.info(f"Roll-out {step} of {self.n_steps_rollout}")
                with torch.no_grad():
                    # Get visual-language-state input from the environment, and optionally apply domain randomization. 
                    batch = self.fetch_batch_from_env(self.obs_venv)  # automatically put on model.device
                    if not self.eval_mode and hasattr(self, 'domain_randomization'): # apply domain randomization during training. 
                        batch = self.domain_randomizer.apply(batch)
                    # Inference by integrating the ODE/SDE
                    action_chunk_venv, chains_venv, log_probs_venv, values_venv = self.policy.select_action(batch, evaluate=self.eval_mode) # type: ignore
                # Interact with the MultiActionWrapper environment
                action_chunk_venv=action_chunk_venv.to(self.sim_device, self.sim_dtype)
                nextobs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = ret_batch = self.venv.step(action_chunk_venv)
                self.done_venv = terminated_venv | truncated_venv # type:ignore
                #################################################
                if self.verbose_buffer_filling: 
                    logger.info(f"DEBUG::PPO_Runner: terminated_venv={terminated_venv}, truncated_venv={truncated_venv}, info_venv['success']={info_venv['success']}, reward_venv={reward_venv}")
                #################################################
                # Log this transition
                if self.verbose_each_step:
                    log_step_simple(step, ret_batch, self.done_venv)
                # Record video data for first episode only
                if self.record_video():
                    self.record_video_frame(info_venv) # log the last action # type:ignore
                # Record the data in the buffer when in training mode, and record it in the eval buffer when in eval mode. 
                if not self.eval_mode:
                    self.buffer.add(step, batch, chains_venv, log_probs_venv, values_venv, reward_venv, terminated_venv, truncated_venv)
                else:
                    self.eval_buffer.add(step, reward_venv, terminated_venv, truncated_venv)
                    if self.verbose_buffer_filling:
                        logger.info(f"DEBUG::PPO_Runner: self.eval_buffer.reward_trajs[step]={self.eval_buffer.reward_trajs[step]}, self.eval_buffer.firsts_trajs[step]={self.eval_buffer.firsts_trajs[step]}")
                # Update the previous observation for the next step and record sample complexity. 
                self.obs_venv = nextobs_venv # type: ignore
                self.cnt_train_step+= self.n_envs * self.act_steps if not self.eval_mode else 0
                
                # Periodic memory cleanup during rollout to prevent OOM during partial resets
                if step % self.memory_cleanup_freq_rollout == 0 and step > 0:  # Cleanup every N steps
                    cleanup_cuda_memory()
            
            # Create video for first episode only
            if self.record_video():
                buffer_to_align = self.eval_buffer if self.eval_mode else self.buffer
                self.save_video(buffer=buffer_to_align)
            # Summarize episode reward statistics
            if not self.eval_mode:
                self.buffer.summarize_episode_reward()
            else:
                self.eval_buffer.summarize_episode_reward()
            # Update training buffer
            if not self.eval_mode:
                self.buffer: PPOFlowImgBufferGPU
                self.buffer.update(self.fetch_batch_from_env(nextobs_venv), self.policy) # type: ignore  # Normalize reward and update the advantage function and returns. The input arguments are used to update the value function V(s_t+1) for the truncated actions in the buffer. 
                self.agent_update(verbose=self.verbose_update)
                # Cleanup after agent update to free gradient computation memory
                cleanup_cuda_memory()
            
            self.log()
            self.update_lr()
            self.adjust_finetune_schedule()# update finetune scheduler to regulate SDE-based denoising. 
            self.save_model()
            
            self.itr += 1
            if self.use_early_stop: self.check_early_stop()
            cleanup_cuda_memory()
    
    def fetch_batch_from_env(self,obs_venv) -> dict[str, Tensor]:
        """Collect the visual, language, and proprioception observations from the environment. 
        Notice that we have to manually fetch the language from the simulator in case envs reset in the middel of an episode. 
        Returns:
            batch:dict[str, Tensor] 
            batch = {
                self.proprioception_key: proprioception,                           # Tensor[B, state_dim]. B=num_envs. 
                self.language_key: language_instruction,                           # A length-B list of strings. 
                self.rgb_keys[0]:  rgb_image_camera_0                              # Images from camera 0: Tensor[B, C, H, W]
                self.rgb_keys[1]:  rgb_image_camera_1                              # Images from camera 1: Tensor[B, C, H, W]
                ...
                self.rgb_keys[num_cameras-1]:  rgb_image_camer_{num_cameras-1}     # Images from the last camera: Tensor[B, C, H, W]
        }
        """
        proprioception: Tensor =self.env_unwrapp.agent.robot.get_qpos().to(self.model_device)                       # qpos (joint angles)
        language_instruction: List[str] = self.env_unwrapp.get_language_instruction()                               # type: ignore a list of strings with len=B=num_envs
        batch = {
            self.proprioception_key: proprioception,            # Tensor[B, state_dim]. B=num_envs. 
            self.language_key: language_instruction,            # A length-B list of strings. 
        }
        # add image lists (possibly supporting multiple cameras, including writst camera and 3rd person view camera)
        rgb_image_list=fetch_rgb_from_obs(obs_venv, self.sim_device, self.model_device) # type: ignore   # [B, C, H, W], where B here is the self.num_envs
        for rgb_key, rgb_image in zip(self.rgb_keys, rgb_image_list):
            batch.update({
                rgb_key: rgb_image                              # Tensor[B, C, H, W], this is transposed from simulator output [B,H,W,C], because PaliGemma image encoder receives images like [B,C,H,W], see code PATH_TO_YOUR_CONDA/envs/pi_r/lib/python3.10/site-packages/transformers/models/paligemma/modeling_paligemma.py function `get_image_features`
            })                        
        if self.verbose_input: logger.info(f"Inputs to pi-zero: proprioception_state={proprioception.shape}, task instructions: {language_instruction}, rgb_images={self.rgb_keys}, rgb_image.shape={[batch[rgb_key].shape for rgb_key in self.rgb_keys]}")
        return batch

    def set_devices(self):
        # model and simulation devices
        sim_backend, sim_device, sim_device_id, model_device=set_model_sim_devices(self.cfg)
        self.sim_backend=sim_backend
        self.sim_device=sim_device
        self.sim_dtype = torch.float32   # the dtype of the simulator environment that determines the dtype of the actions that can be directly deployed to the environment. 
        self.sim_device_id=sim_device_id
        
        self.model_device=model_device
        self.model_dtype = torch.float32 # the dtype of the model that determines the dtype of the actions that can be directly deployed to the environment. 
        # buffer devices
        self.buffer_device = torch.device(self.cfg.buffer.device)
        self.buffer_dtype = getattr(torch, self.cfg.buffer.dtype)
        
    def load_model(self):
        logger.info(f"Loading dataset statistics (normalization) from {self.normalization_path}")
        self.dataset_stats = torch.load(self.normalization_path, map_location='cpu')
        
        logger.info(f"Loading model from {self.model_path}")
        self.policy, self.model_config= create_model(self.cfg, self.dataset_stats, self.model_device)
        
        logger.info(f"Successfully loaded model on {self.model_device}")
        save_model_config(self.model_config, self.log_dir, logger)
        save_model_architecture(self.policy, log_dir=Path(self.log_dir))
    
    def save_model(self):
        if self.itr % self.save_model_freq ==0 and self.itr >0:
            save_rlft_model_best(checkpoint_dir=Path(self.log_dir),
                                itr=self.itr,
                                cnt_train_step=self.cnt_train_step, 
                                is_current_best=self.is_current_best,
                                model=self.policy,
                                actor_optimizer=self.actor_optimizer,
                                actor_scheduler=self.actor_lr_scheduler,
                                critic_optimizer=self.critic_optimizer,
                                critic_scheduler=self.critic_lr_scheduler)
    
    def resume_training_state(self):
        """Recover the itr, cnt_train_step, optimizer, learning rate, and optionally the noise schedulers from self.resume_dir/training_state/. 
        Before calling this function, we should make sure that the model in the same directory is already loaded. This function
        only loads the training state to reumse RLFT training. 
        
        resume_dir-----------------------------------------------------------------------------------------> cfg.resume_dir
             ├── model/------------------------------------------------------------------------------------> cfg.model.path
             │   ├── config.json  # policy config (fixed to remove 'type' field and normalize device)
             │   ├── model.safetensors  # policy weights
             |__ training_state/---------------------------------------------------------------------------> training state to resume in thie function call.
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
        logger.info(f"Resuming training from {self.resume_dir}")
        if not self.resume_dir:
            raise ValueError(f"resume_dir is None, please check your configuration if you want to resume from previous checkpoint.")
        if not os.path.exists(self.resume_dir):
            raise ValueError(f"resume_dir {self.resume_dir} does not exists, please check your configuration if you want to resume from prev")
        if not os.path.isdir(self.resume_dir):
            raise ValueError(f"resume_dir {self.resume_dir} is not a directory, please check your configuration if you want to resume from prev")

        # Read from rlft_metadata.json
        with open(os.path.join(self.resume_dir,"rlft_metadata.json"), "r") as f:
            metadata = json.load(f)

        # Load training stage information. 
        self.itr = metadata["itr"]
        self.n_train_itr += self.itr # train for another xx iters.
        self.cnt_train_step = metadata["cnt_train_step"]
        logger.info(f"Resume training from itr={self.itr}, total train steps={self.cnt_train_step}.")
        
        # load optimizers
        actor_state_dir=os.path.join(self.resume_dir, 'training_state', 'actor')
        if os.path.isdir(actor_state_dir):
            self.actor_optimizer=load_optimizer_state(self.actor_optimizer, Path(actor_state_dir))
            logger.info(f"Successfully loaded actor optimizers from {actor_state_dir}")
        else:
            raise ValueError(f"actor state directory {actor_state_dir} is not valid. Please check your config or data saving logic.")
        
        critic_state_dir=os.path.join(self.resume_dir, 'training_state', 'critic')
        if os.path.isdir(critic_state_dir):
            self.critic_optimizer=load_optimizer_state(self.critic_optimizer, Path(critic_state_dir))
            logger.info(f"Successfully loaded critic optimizers from {critic_state_dir}")
        else:
            raise ValueError(f"actor state directory {critic_state_dir} is not valid. Please check your config or data saving logic.")
        
        # load learning rate schedulers
        if os.path.exists(os.path.join(actor_state_dir, 'scheduler_state.json')):
            self.actor_lr_scheduler=load_scheduler_state(self.actor_lr_scheduler, Path(actor_state_dir))
            logger.info(f"Successfully loaded actor learning rate scheduler from {actor_state_dir}")
        else:
            logger.warning(f"No actor learning rate scheduler found in path {actor_state_dir}, resume from newly calibrated scheduler.")
            for _ in range(self.itr): # recover lr schedulers from scratch if no learning rate scheduler found. 
                self.actor_lr_scheduler.step()

        if os.path.exists(os.path.join(critic_state_dir, 'scheduler_state.json')):
            self.critic_lr_scheduler=load_scheduler_state(self.critic_lr_scheduler, Path(critic_state_dir))
            logger.info(f"Successfully loaded critic learning rate scheduler from {critic_state_dir}")
        else:
            logger.warning(f"No critic learning rate scheduler found in path {critic_state_dir}, resume from newly calibrated scheduler.")
            for _ in range(self.itr): # recover lr schedulers from scratch if no learning rate scheduler found. 
                self.critic_lr_scheduler.step()
        
        # load noise schedulers (optionally)
        # for reinflow
        if self.policy.noise_scheduler_type == 'const':
            updated_noise_logvar_range=[
                self.cfg.rlft_config.explore_noise_net.noise_logvar_min, 
                self.cfg.rlft_config.explore_noise_net.noise_logvar_max
            ]
            self.policy.model.reinflow_explore_noise_net.set_noise_range(updated_noise_logvar_range)
            logger.info(f"Updated noise_logvar_range={updated_noise_logvar_range} (self.policy.noise_scheduler_type={self.policy.noise_scheduler_type})")   
    
    def create_env(self):
        from scripts.env.multi_action_wrapper import MultiActionWrapper
        from scripts.env.per_step_reward_wrapper import PerStepRewardWrapper
        from functools import partial
        
        self.venv: ManiSkillVectorEnv=setup_maniskill_env(env_id=self.env_id, 
                                                     num_envs=self.n_envs, 
                                                     max_episode_len=self.n_steps_episode, 
                                                     sim_backend=self.sim_backend, 
                                                     sim_device=self.sim_device, 
                                                     sim_device_id=self.sim_device_id, 
                                                     sim_config=self.sim_config, sensor_config=self.sensor_config, 
                                                     obs_mode=self.obs_mode, control_mode=self.control_mode, 
                                                     episode_mode='train',
                                                     wrappers=[
                                                        partial(PerStepRewardWrapper),
                                                        partial(MultiActionWrapper),
                                                    ])# be very, very carefule about the order of the wrappers !!
        
        self.env_unwrapp: BaseEnv=self.venv.unwrapped # env.reward_mode==None, but it provides episodic success information in info['success']
    
    def build_optimizer_scheduler(self):
        """
        Currently the critic model is very very shallow so we share the optimizer for the actor, critic, and potentially noise net. 
        """
        self.params_actor = []
        self.params_critic = []
        for name, param in self.policy.named_parameters():
            if param.requires_grad:
                if "critic"in name:
                    self.params_critic.append(param)
                else:
                    self.params_actor.append(param)
        if len(self.params_actor)==0:
            raise ValueError(f"No actor parameters found in the model. Please make sure the current parameter extraction script matches your model architecture and naming convention.")
        if len(self.params_critic)==0:
            raise ValueError(f"No critic parameters found in the model. Please make sure the current parameter extraction script matches your model architecture and naming convention.")
        self.actor_optimizer=torch.optim.AdamW(self.params_actor, lr=self.actor_lr, betas=tuple(self.actor_betas), eps=self.actor_eps, weight_decay=self.actor_weight_decay)
        self.critic_optimizer=torch.optim.AdamW(self.params_critic, lr=self.critic_lr, betas=tuple(self.critic_betas), eps=self.critic_eps, weight_decay=self.critic_weight_decay)
        logger.info(colored(f"✓ Successfully built actor and critic optimizers with lr={self.actor_lr} and {self.critic_lr}.", "green", "on_green"))
        
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=self.cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=self.cfg.train.actor_lr_scheduler.cycle_mult,
            max_lr=self.cfg.train.actor_optimizer.lr,
            min_lr=self.cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=self.cfg.train.actor_lr_scheduler.warmup_steps,
            max_lr_decrease_per_cycle=self.cfg.train.actor_lr_scheduler.max_lr_decrease_per_cycle,
        )
        self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_optimizer,
            first_cycle_steps=self.cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=self.cfg.train.critic_lr_scheduler.cycle_mult,
            max_lr=self.cfg.train.critic_optimizer.lr,
            min_lr=self.cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=self.cfg.train.critic_lr_scheduler.warmup_steps,
            max_lr_decrease_per_cycle=self.cfg.train.critic_lr_scheduler.max_lr_decrease_per_cycle,
        )
        logger.info(colored(f"✓ Successfully built actor and critic learning rate schedulers.", "green", "on_green"))
        self.visualize_lr()
        
    def build_noise_scheduler(self):
        if self.policy.model.sde_mode == 'flow-grpo':
            # Regulate the std of hard-coded noise. 
            if self.policy.noise_scheduler_type == 'constant':
                self.noise_scheduler = lambda x: self.policy.model.noise_level
            else:
                raise NotImplementedError(f"'flow-grpo' noise_scheduler_type={self.policy.noise_scheduler_type} is not currently supported.")
        elif self.policy.model.sde_mode=='reinflow':
            # Regulte the upper and lower bounds of the learnable noise. 
            noise_scheduler_type_options=['constant_schedule_itr', 'learn_decay', 'const', 'learn']
            if self.policy.noise_scheduler_type == 'constant_schedule_itr':
                self.noise_scheduler = get_scheduler(schedule_type='cosine_warmup',
                                                                min=0.016,
                                                                warmup_steps=self.n_train_itr * 0.01,
                                                                max=0.08, #0.15,
                                                                hold_steps=self.n_train_itr * 0.29,
                                                                anneal_steps=self.n_train_itr * 0.7)
                
                explore_noises = [self.noise_scheduler(t) for t in np.arange(self.n_train_itr)]
                plt.figure()
                plt.plot(np.arange(self.n_train_itr), explore_noises)
                name=os.path.join(self.log_dir,'explore_noise')+'.png'
                plt.savefig(name)
                plt.close()
                logger.info("Exploration noise saved to %s" % name)
            elif self.policy.noise_scheduler_type == 'learn_decay':
                max_std=self.cfg.rlft_config.explore_noise_net.noise_logvar_max
                min_std=self.cfg.rlft_config.explore_noise_net.noise_logvar_min
                self.max_noise_decay_ratio=self.cfg.train.get('max_noise_decay_ratio', 0.7)
                max_std_decayed=min_std*(1-self.max_noise_decay_ratio)+max_std*self.max_noise_decay_ratio
                self.max_noise_hold_ratio=self.cfg.train.get('max_noise_hold_ratio', 0.35)
                self.noise_scheduler = get_scheduler(schedule_type='cosine',
                                                                max=max_std,
                                                                hold_steps=self.n_train_itr * self.max_noise_hold_ratio,
                                                                anneal_steps=self.n_train_itr * (1-self.max_noise_hold_ratio),
                                                                min=max_std_decayed)
                max_explore_noises = [self.noise_scheduler(t) for t in np.arange(self.n_train_itr)]
                min_explore_noises = [min_std for _ in np.arange(self.n_train_itr)]
                plt.figure()
                plt.plot(np.arange(self.n_train_itr), max_explore_noises, label=f'max_std:{max_std:.2f} to {max_std_decayed:.2f}')
                plt.plot(np.arange(self.n_train_itr), min_explore_noises, label=f'min_std:{min_std:.2f}')
                plt.legend()
                name=os.path.join(self.log_dir,'explore_noise')+'.png'
                plt.savefig(name)
                plt.close()
                logger.info("Exploration noise level bounds saved to %s" % name)
            elif self.policy.noise_scheduler_type in noise_scheduler_type_options:
                max_std=self.cfg.rlft_config.explore_noise_net.noise_logvar_max
                min_std=self.cfg.rlft_config.explore_noise_net.noise_logvar_min
                max_explore_noises = [max_std for _ in np.arange(self.n_train_itr)]
                min_explore_noises = [min_std for _ in np.arange(self.n_train_itr)]
                logger.info(f"Received self.policy.noise_scheduler_type={self.policy.noise_scheduler_type}, will use constant noise ranges [{min_std:.2f}, {max_std:.2f}]")
                plt.figure()
                plt.plot(np.arange(self.n_train_itr), max_explore_noises, label=f'max_std:{max_std:.2f}')
                plt.plot(np.arange(self.n_train_itr), min_explore_noises, label=f'min_std:{min_std:.2f}')
                plt.legend()
                name=os.path.join(self.log_dir,'explore_noise')+'.png'
                plt.savefig(name)
                plt.close()
                logger.info("Exploration noise level bounds saved to %s" % name)
            else:
                raise ValueError(f"Invalid noise scheduler type: {self.policy.noise_scheduler_type}. Currently we only support {noise_scheduler_type_options}. Please check the model.noise_scheduler_type in the model config.")
    
    def build_domain_randomization(self):
        self.domain_randomizer = DomainRandomization(self.domain_rand_cfg)
    
    def create_buffer(self):
        logger.info(f"Initializing buffer on {self.buffer_device}")
        self.buffer = PPOFlowImgBufferGPU(
            device=self.buffer_device,
            dtype=self.buffer_dtype,
            n_rollout_steps=self.n_steps_rollout,
            n_envs=self.n_envs,
            n_denoising_steps= self.n_denoising_steps, 
            n_action_steps=self.n_action_steps,
            act_steps=self.act_steps,
            max_action_dim=self.policy.model.config.max_action_dim,
            n_cond_step=self.n_cond_step,
            visuomotor_obs_dict=self.visuomotor_obs_dict,
            language_key=self.language_key,
            reward_scale_const=self.reward_scale_const,
            reward_scale_running=self.reward_scale_running,
            ignore_nextvalue_when_truncated=self.ignore_nextvalue_when_truncated,
            success_rew_threshold_in_chunk=self.success_rew_threshold_in_chunk,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        logger.info(colored(f"✓ Successfully created training buffer: {self.buffer.__class__} on {self.buffer_device} with dtype {self.buffer_dtype}", "green", "on_green"))
        
        self.eval_buffer = PPOEvalBufferGPU(
            device=self.buffer_device,
            dtype=self.buffer_dtype,
            n_rollout_steps=self.n_steps_rollout,
            act_steps=self.act_steps,
            n_envs=self.n_envs,
            success_rew_threshold_in_chunk=self.success_rew_threshold_in_chunk
        )
        logger.info(colored(f"✓ Successfully created evaluation buffer: {self.eval_buffer.__class__} on {self.buffer_device} with dtype {self.buffer_dtype}", "green", "on_green"))
        
    def set_model_mode(self):
        # Define train or eval - all envs restart
        if self.skip_initial_eval and self.itr ==0:
            self.eval_mode = False 
        else:
            if self.resume:
                self.eval_mode = True
                self.resume = False
            else:
                self.eval_mode = self.itr % self.val_freq == 0
        self.policy.eval() if self.eval_mode else self.policy.train()
        self.last_itr_eval = self.eval_mode
    
    def prepare_run(self):
        # Start training loop
        self.timer = Timer()
        self.run_results = []
        self.cnt_train_step = 0
        self.last_itr_eval = False
        self.done_venv = torch.zeros(self.n_envs, dtype=self.buffer_dtype, device=self.buffer_device)

    def visualize_lr(self):
        steps = []
        actor_lrs = []
        critic_lrs = []
        for step in range(self.cfg.train.n_train_itr):
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            steps.append(step)
            actor_lrs.append(self.actor_optimizer.param_groups[0]["lr"])
            critic_lrs.append(self.critic_optimizer.param_groups[0]["lr"])
        plt.subplot(1,2,1)
        plt.plot(steps, actor_lrs, label='actor', color = 'blue')
        plt.legend(loc='upper right')
        plt.subplot(1,2,2)
        plt.plot(steps, critic_lrs,label='critic', color='red')
        plt.legend(loc='upper right')
        lr_save_path = os.path.join(self.log_dir, 'test_lr_schedulers.png')
        plt.savefig(lr_save_path)
        logger.info(f"learning rate saved to {lr_save_path}")
        plt.close()
        if isinstance(self.actor_lr_scheduler, CustomScheduler):
            self.actor_lr_scheduler.reset()
        if isinstance(self.critic_lr_scheduler, CustomScheduler):
            self.critic_lr_scheduler.reset()
    
    def check_early_stop(self):
        """
        When RLFT fails (success rate less than 5%), stop training process early to prevent wasting compute. 
        """
        if not self.eval_mode: 
            if self.buffer.success_rate < 0.05 or self.buffer.avg_episode_reward < 0.05:
                logger.info(f"Your finetuning failed. success_rate={self.buffer.success_rate*100:.2f}% and avg_episode_reward={self.buffer.avg_episode_reward:.2f}")
                exit()
    
    def reset_env_model(self):
        buffer_device=self.buffer_device
        # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
        if self.reset_at_iteration or self.eval_mode or self.last_itr_eval:
            env_seeds = self.seed if self.same_seed_for_all_envs else [self.seed+i for i in range(self.n_envs)]
            self.obs_venv, self.info= self.venv.reset(seed=env_seeds,  options=self.reset_options_venv)
            self.buffer.firsts_trajs[0] = torch.ones((self.n_envs), dtype=self.buffer_dtype, device=self.buffer_device)
            self.eval_buffer.firsts_trajs[0] = torch.ones((self.n_envs), dtype=self.buffer_dtype, device=self.buffer_device)
            # Cleanup after environment reset to prevent OOM during partial resets
            cleanup_cuda_memory()
        else:
            # if done at the end of last iteration, the envs are just reset
            if buffer_device == 'cpu':
                self.buffer.firsts_trajs[0] = self.done_venv.cpu().numpy()  # type: ignore
                self.eval_buffer.firsts_trajs[0] = self.done_venv.cpu().numpy() # type: ignore
            else:
                self.buffer.firsts_trajs[0] = self.done_venv  # type: ignore
                self.eval_buffer.firsts_trajs[0] = self.done_venv  # type: ignore
            # Note: Partial resets from env.step() are automatically handled by ActionQueueManager
            # No manual action queue management needed here
    
    def update_lr(self):
        if self.target_kl and self.lr_schedule == 'adaptive_kl':   # adapt learning rate according to kl divergence on each minibatch.
            return # don't update lr for now, update in the minibatch generator instead.  
        else:
            self.critic_lr_scheduler.step()  # Always step critic scheduler  
            if self.itr >= self.n_critic_warmup_itr: 
                self.actor_lr_scheduler.step()  # Only step actor scheduler after warmup
            logger.info(f"""learning rate updated. actor_lr={self.actor_optimizer.param_groups[0]["lr"]:.2e}, critic_lr={self.critic_optimizer.param_groups[0]["lr"]:.2e}""")
    
    def update_lr_adaptive_kl(self, approx_kl):
        """
        Note: this is not used in the current implementation. 
        """
        min_actor_lr = 1e-5
        max_actor_lr = 5e-4
        
        min_critic_lr = 1e-5
        max_critic_lr = 1e-3
        tune='maintains'
        if approx_kl > self.target_kl * 2.0:
            self.actor_lr = max(min_actor_lr, self.actor_lr / 1.5)
            self.critic_lr = max(min_critic_lr, self.critic_lr / 1.5)
            tune = 'decreases'
        elif 0.0 < approx_kl and approx_kl < self.target_kl / 2.0:
            self.actor_lr = min(max_actor_lr, self.actor_lr * 1.5)
            self.critic_lr = min(max_critic_lr, self.critic_lr * 1.5)
            tune = 'increases'
        self.actor_optimizer.param_groups[0]["lr"] = self.actor_lr
        self.critic_optimizer.param_groups[0]["lr"] = self.critic_lr
        logger.info(f"""adaptive kl {tune} lr: actor_lr={self.actor_optimizer.param_groups[0]["lr"]:.2e}, critic_lr={self.critic_optimizer.param_groups[0]["lr"]:.2e}""")
    
    def adjust_finetune_schedule(self):
        if self.policy.model.sde_mode == 'sde':
            self.policy.model.noise_level = self.noise_scheduler(self.itr)
        elif self.policy.model.sde_mode == 'reinflow':
            # constant noise levels in denoising steps, but the level changes with training iterations. 
            if self.policy.noise_scheduler_type == 'const_schedule_itr':
                explore_noise_std = self.noise_scheduler(self.itr)
                self.policy.model.set_logprob_noise_levels(force_level=explore_noise_std)
            # gradually decrease the noise upper bound, to prevent noisy samples from hurting the model. 
            if self.policy.noise_scheduler_type == 'learn_decay':
                updated_noise_std_range=[
                    self.policy.model.reinflow_noise_logvar_min,   # keep lower bound.
                    self.noise_scheduler(self.itr)                 # schedule upper bound. 
                ]
                self.policy.model.reinflow_explore_noise_net.set_noise_range(updated_noise_std_range)
                logger.info(f"Updated noise_std_range={updated_noise_std_range} (self.policy.noise_scheduler_type={self.policy.noise_scheduler_type})")
        else:
            raise ValueError(f"Invalid SDE type: {self.policy.model.sde_mode}. Please check the model.sde_mode in the model config.")
    
    
    def minibatch_generator(self):
        self.approx_kl = 0.0
        
        obs, chains, returns, oldvalues, advantages, oldlogprobs =  self.buffer.make_dataset()
        # Explained variation of future rewards using value function
        self.explained_var = self.buffer.get_explained_var(oldvalues, returns)
        # define Q values as the old Q values to align with the definition in diffusion ppo. you can change those back to new Q values 
        self.Q_values = oldvalues.mean().item()
        
        self.total_steps = self.n_steps_rollout * self.n_envs  
        for update_epoch in range(self.update_epochs):
            self.kl_change_too_much = False
            indices = torch.randperm(self.total_steps, device=self.buffer_device)
            for batch_id, start in enumerate(range(0, self.total_steps, self.batch_size)):
                end = start + self.batch_size
                inds_b = indices[start:end]                            # b is for batchsize
                
                chains_mb = chains[inds_b].to(device=self.model_device, dtype=self.model_dtype)
                chains_pre = chains_mb[:, :-1]
                chains_next = chains_mb[:, 1:]
                
                minibatch = (
                    {k: v[inds_b].to(device=self.model_device, dtype=self.model_dtype) if isinstance(v, Tensor)       # visuomotor (tensor) obs
                     else [v[i] for i in inds_b]                                                                      # language (lists of strings) obs
                     for k, v in obs.items()},
                    chains_pre,                                                                                       # chains_prev_b self.batch_size x self.horizon_steps x self.act_dim 
                    chains_next,                                                                                      # chains_next_b self.batch_size x self.horizon_steps x self.act_dim 
                    returns[inds_b].to(device=self.model_device, dtype=self.model_dtype),                             # returns_b        [B,] 
                    oldvalues[inds_b].to(device=self.model_device, dtype=self.model_dtype),                           # values_b         [B,] 
                    advantages[inds_b].to(device=self.model_device, dtype=self.model_dtype),                          # advantages_b     [B,]  there are many duplicated entries.
                    oldlogprobs[inds_b].to(device=self.model_device, dtype=self.model_dtype)                          # logprobs_b       self.batch_size x self.denoising_steps x self.horizon_steps x self.act_dim 
                )
                
                if self.lr_schedule=='fixed' and self.target_kl and self.approx_kl > self.target_kl: # we can also use adaptive KL instead of early stopping.
                    self.kl_change_too_much = True
                    logger.warning(f"KL change too much, approx_kl ={self.approx_kl} > {self.target_kl} = target_kl, stop optimization.")
                    break
                
                yield update_epoch, batch_id, minibatch    
    
    def agent_update(self, verbose=True):
        clipfracs_list = []
        noise_std_list = []
        actor_norm=0.0
        critic_norm=0.0
        for update_epoch, batch_id, minibatch in self.minibatch_generator():
            # minibatch gradient descent
            pg_loss, entropy_loss, v_loss, bc_loss, \
                clipfrac, approx_kl, ratio, \
                    oldlogprob_min, oldlogprob_max, oldlogprob_std, \
                        newlogprob_min, newlogprob_max, newlogprob_std, \
                            noise_std= self.policy.loss(*minibatch, 
                                                        use_bc_loss=self.use_bc_loss, 
                                                        bc_loss_type=self.bc_loss_type, 
                                                        clip_intermediate_actions=self.clip_intermediate_actions,
                                                        verbose=verbose)
            self.approx_kl = approx_kl
            if verbose:
                logger.info(f"update_epoch={update_epoch}/{self.update_epochs}, batch_id={batch_id}/{max(1, self.total_steps // self.batch_size)}, ratio={ratio:.3f}, clipfrac={clipfrac:.3f}, approx_kl={self.approx_kl:.2e}")
            
            if update_epoch ==0  and batch_id ==0 and np.abs(ratio-1.00)> self.initial_ratio_error_threshold:
                logger.info(f"Warning: ratio={ratio} not 1.00 when update_epoch ==0  and batch_id ==0, there must be some bugs in your code!")
            
            if self.target_kl and self.lr_schedule == 'adaptive_kl':
                self.update_lr_adaptive_kl(self.approx_kl)
            
            loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_coeff
            
            clipfracs_list += [clipfrac]
            noise_std_list += [noise_std]
            
            loss.backward()
            if (batch_id + 1) % self.grad_accumulate == 0:
                # debug the losses
                actor_norm = torch.nn.utils.clip_grad_norm_(self.params_actor, max_norm=float('inf'))
                critic_norm = torch.nn.utils.clip_grad_norm_(self.params_critic, max_norm=float('inf'))
                if verbose: logger.info(f"before clipping: actor_norm={actor_norm:.2e}, critic_norm={critic_norm:.2e}")
                # Add checkpoint to test if crtic wawrmup is sufficient:
                if self.itr == self.n_critic_warmup_itr:
                    logger.info(f"Critic warmup finished with {self.itr}")
                    if self.explained_var < 0.6:
                        raise ValueError(colored(f"Explained variance is too low at iteration {self.itr}, consider increasing n_critic_warmup_itr or decreasing learning rate, say, by half."), "red", "on_red")
                # update actor: after critic warmup update the actor less frequently but more times. 
                if self.itr >= self.n_critic_warmup_itr:
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.params_actor, self.max_grad_norm)
                    self.actor_optimizer.step()
                else:
                    logger.info(f"Skip actor update during critic warmup iteration {self.itr}/{self.n_critic_warmup_itr}")
                
                # update critic
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.params_critic, self.max_grad_norm)
                self.critic_optimizer.step()
                
                # release gradient accumulation
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                # report
                logger.info(f"run grad update at batch {batch_id}, approx_kl: {approx_kl:.3e}, update_epoch: {update_epoch}/{self.update_epochs}, num_batch: {self.total_steps //self.batch_size}")
                
                # Cleanup after gradient updates to prevent memory accumulation
                if batch_id % self.memory_cleanup_freq_grad == 0:  # Cleanup every N gradient updates
                    cleanup_cuda_memory()
        
        clip_fracs=np.mean(clipfracs_list)
        noise_stds=np.mean(noise_std_list)
        self.train_ret_dict = {
                "loss": loss,
                "pg loss": pg_loss,
                "value loss": v_loss,
                "entropy_loss": entropy_loss,
                "bc_loss": bc_loss,
                "approx kl": self.approx_kl,
                "ratio": ratio,
                "clipfrac": clip_fracs,
                "explained variance": self.explained_var,
                "old_logprob_min": oldlogprob_min,
                "old_logprob_max": oldlogprob_max,
                "old_logprob_std": oldlogprob_std,
                "new_logprob_min": newlogprob_min,
                "new_logprob_max": newlogprob_max,
                "new_logprob_std": newlogprob_std,
                "actor_norm": actor_norm,
                "critic_norm": critic_norm,
                "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
                "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
                "noise_std": noise_stds,
                "Q_values": self.Q_values   # define Q values as the old Q values to align with the definition in diffusion ppo. you can change those back to new Q values 
            }
        if self.policy.model.sde_mode =='reinflow':
            self.train_ret_dict.update({"reinflow_noise_logvar_min": self.policy.model.reinflow_noise_logvar_min,
                                        "reinflow_noise_logvar_max": self.policy.model.reinflow_noise_logvar_max})
    
    def log(self, train_prt_str_additional="", train_log_dict_additional={}):
        '''
        train_prt_str_additional: str, additional information in training that will be logged to console that is not included in train_prt_str_basic
        train_log_dict_additional: dict, additional information in training that will be logged to wandb that is not included in train_log_dict_basic
        '''
        BOLDSTART = '\033[1m'
        BOLDEND = '\033[0m'

        self.run_results.append(
            {
                "itr": self.itr,
                "step": self.cnt_train_step,
            }
        )
        if self.save_trajs:
            raise NotImplementedError("Saving trajectories is not implemented yet.")
        if self.itr % self.log_freq == 0:
            time = self.timer()
            self.run_results[-1]["time"] = time
            if self.eval_mode:
                # Updated evaluation log with avg ± std formatting
                logger.info(create_bordered_text(
                    f"{BOLDSTART}Evaluation at itr {self.itr}{BOLDEND}:\n"
                    f"Model: {self.policy.__class__.__name__}\n"
                    f"Environment: {self.env_id} x {self.n_envs}\n"
                    f"Num denoising steps: {self.n_denoising_steps}\n"
                    f"Seed: {self.seed}\n"
                    f"Success Rate: {self.eval_buffer.success_rate * 100:3.2f}% ± {self.eval_buffer.std_success_rate * 100:3.2f}%\n"
                    f"Episode Reward: {self.eval_buffer.avg_episode_reward:8.2f} ± {self.eval_buffer.std_episode_reward:8.2f}\n"
                    f"Best Reward (per action): {self.eval_buffer.avg_best_reward:8.2f} ± {self.eval_buffer.std_best_reward:8.2f}\n"
                    f"Episode Length: {self.eval_buffer.avg_episode_length:8.2f} ± {self.eval_buffer.std_episode_length:8.2f}\n"
                    f"Num Episode Finished: {self.eval_buffer.n_episode_finished}\n"
                    f"Num Successful Episodes: {self.eval_buffer.n_successful_episodes}\n"
                    f"actor_lr :{self.actor_optimizer.param_groups[0]['lr']:.2e}\n"
                    f"critic_lr :{self.critic_optimizer.param_groups[0]['lr']:.2e}\n"
                ))
                eval_dict={
                            "eval/success rate": self.eval_buffer.success_rate,
                            "eval/avg episode reward": self.eval_buffer.avg_episode_reward,
                            "eval/avg best reward": self.eval_buffer.avg_best_reward,
                            "eval/avg episode length": self.eval_buffer.avg_episode_length,
                            "eval/num episode": self.eval_buffer.n_episode_finished,
                            "eval/num successful episodes": self.eval_buffer.n_successful_episodes,
                            "eval/std success rate": self.eval_buffer.std_success_rate,
                            "eval/std episode reward": self.eval_buffer.std_episode_reward,
                            "eval/std best reward": self.eval_buffer.std_best_reward,
                            "eval/std episode length": self.eval_buffer.std_episode_length,
                    }
                # convert everything to floating points
                for key, value in eval_dict.items():
                    if isinstance(value, Tensor):
                        eval_dict[key]=value.item()
                self.run_results[-1].update(eval_dict)
                if self.use_wandb:
                    wandb.log(
                        data=eval_dict,
                        step=self.itr,
                        commit=False,
                    )
                if self.current_best_reward < self.eval_buffer.avg_episode_reward:
                    self.current_best_reward = self.eval_buffer.avg_episode_reward
                    self.is_current_best = True
                    logger.info(f"New best reward evaluated: {self.current_best_reward:4.3f}")
            else:
                # Updated training log with avg ± std formatting
                train_prt_str_basic = (
                    f"{BOLDSTART}Training at itr {self.itr}{BOLDEND}:\n"
                    f"Total Step {self.cnt_train_step / 1e6:4.3f} M \n"
                    f"Time: {time:8.3f}\n"
                    f"Env: {self.env_id} x {self.n_envs}\n"
                    f"Episode Reward: {self.buffer.avg_episode_reward:8.2f} ± {self.buffer.std_episode_reward:8.2f}\n"
                    f"Success Rate: {self.buffer.success_rate * 100:3.2f}% ± {self.buffer.std_success_rate * 100:3.2f}% \n"
                    f"Avg Best Reward: {self.buffer.avg_best_reward:8.2f} ± {self.buffer.std_best_reward:8.2f}\n"
                    f"Episode Length: {self.buffer.avg_episode_length:8.2f} ± {self.buffer.std_episode_length:8.2f}\n"
                    f"Num Episode Finished: {self.buffer.n_episode_finished}\n"
                    f"Num Successful Episodes: {self.buffer.n_successful_episodes}\n"
                    f"Actor lr :{self.actor_optimizer.param_groups[0]['lr']:.2e}\n"
                    f"Critic lr: {self.critic_optimizer.param_groups[0]['lr']:.2e}\n"
                )
                formatted_items = [f"{key}: {value:.3e}" for key, value in self.train_ret_dict.items()]
                num_items_per_row = 1
                for i in range(0, len(formatted_items), num_items_per_row):
                    train_prt_str_basic += " | ".join(formatted_items[i:i+num_items_per_row]) + "\n"
                logger.info(create_bordered_text(train_prt_str_basic + train_prt_str_additional))
                
                # upload to wandb
                train_log_dict_basic = {
                    "train/total env step": self.cnt_train_step,
                    "train/success rate": self.buffer.success_rate,
                    "train/avg episode reward": self.buffer.avg_episode_reward,
                    "train/avg episode length": self.buffer.avg_episode_length,
                    "train/num episode": self.buffer.n_episode_finished,                    
                    "train/num successful episodes": self.buffer.n_successful_episodes,
                    "train/std success rate": self.buffer.std_success_rate,
                    "train/avg best reward": self.buffer.avg_best_reward,
                    "train/std episode reward": self.buffer.std_episode_reward,
                    "train/std best reward": self.buffer.std_best_reward,
                    "train/std episode length": self.buffer.std_episode_length,
                    "train/actor lr": self.actor_optimizer.param_groups[0]["lr"],
                    "train/critic lr": self.critic_optimizer.param_groups[0]["lr"],
                }
                loss_dict = {"loss/"+key: value for key, value in self.train_ret_dict.items()}
                train_log_dict_basic.update(loss_dict)
                train_log_dict = {**train_log_dict_basic, **(train_log_dict_additional or {})}
                # convert everything to floating points
                for key, value in train_log_dict.items():
                    if isinstance(value, Tensor):
                        train_log_dict[key]=value.item()
                self.run_results[-1].update(train_log_dict)
                
                # Log training metrics to WandB
                if self.use_wandb:
                    wandb.log(
                        data=train_log_dict,
                        step=self.itr,
                        commit=True,
                    )
            with open(self.result_path, "wb") as f:
                pickle.dump(self.run_results, f)
        