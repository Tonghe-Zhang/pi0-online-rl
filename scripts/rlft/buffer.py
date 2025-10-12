
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

import numpy as np
import torch
import logging
log = logging.getLogger(__name__)
from termcolor import colored
from utils.reward_scaling_ts import RunningRewardScalerTensor
from scripts.rlft.policy.p0_ppo import PI0PolicyPPO # for update_adv_returns with runner.get_value
# from model.common.critic import ViTCritic

# This buffer needs a lot of changes. It is determined by step rewards but sadly there is no step rewards for spare reward settings. 
# only episode success rates are available. 

# Another thing to consider is the values at denoising steps: are they necessary? 

class PPOEvalBufferGPU:
    def __init__(self,
                 device, 
                 dtype,
                 n_rollout_steps,
                 act_steps,
                 n_envs, 
                 success_rew_threshold_in_chunk
                 ):
        """
        Simple buffer that only stores the reward and the done flags of the episodes. 
        This is used for evaluation only. 
        """
        # Device and dtype of the stored data
        self.buffer_device = device
        self.buffer_dtype = dtype
        
        # Number of rollout steps in each iteration and number of parallel environments
        self.n_rollout_steps = n_rollout_steps
        self.act_steps=act_steps
        self.n_envs = n_envs
        
        # The lowest aggregated reward in the chunk that indicates their is a success in action execution. This depends on the chunk size, aggregation method, and reward scale. 
        self.success_rew_threshold_in_chunk=success_rew_threshold_in_chunk
    
    def reset(self):
        # Store tensor observations (visual + proprioception)
        self.reward_trajs = torch.zeros((self.n_rollout_steps, self.n_envs), dtype=self.buffer_dtype, device=self.buffer_device)
        self.firsts_trajs = torch.zeros((self.n_rollout_steps + 1, self.n_envs), dtype=self.buffer_dtype, device=self.buffer_device)
        self.firsts_trajs[0]=1

    def add(self, rollout_step, reward_venv, terminated_venv, truncated_venv):
        """Add tensors to buffer at the given step.
        Args:
            step: Current step index
            reward_venv: Reward tensor
            terminated_venv: Termination tensor
            truncated_venv: Truncation tensor
        """
        self.reward_trajs[rollout_step] = reward_venv.to(dtype=self.buffer_dtype, device=self.buffer_device)
        self.firsts_trajs[rollout_step + 1] = (terminated_venv | truncated_venv).to(dtype=self.buffer_dtype, device=self.buffer_device)  # done_venv
    
    @torch.no_grad
    def summarize_episode_reward(self):
        """
        Summarize the episode reward and other metrics. This function will be overloaded by the child class 
        to ensure evaluation consistency. 
        """
        episodes_start_end = []
        self.episode_success_flags = []  # New list to store success flags
        # Convert firsts_trajs to numpy for processing
        firsts_trajs_np = self.firsts_trajs.cpu().numpy()  

        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs_np[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))
        
        if len(episodes_start_end) > 0:
            # Select reward_trajs using numpy slicing
            reward_trajs_split = [
                self.reward_trajs[start:end + 1, env_ind].cpu().numpy()
                for env_ind, start, end in episodes_start_end
            ]
            self.n_episode_finished = len(reward_trajs_split)
            
            # Calculating episode_reward using numpy
            episode_reward = np.array([np.sum(reward_traj) for reward_traj in reward_trajs_split])
            
            episode_best_reward_in_chunk = np.array(
                [
                    np.max(reward_traj) 
                    for reward_traj in reward_trajs_split
                ]
            )
            
            success_flags = episode_best_reward_in_chunk >= self.success_rew_threshold_in_chunk
            self.episode_success_flags = success_flags.tolist()  # Store for alignment
            
            # Compute metrics
            self.avg_episode_reward = np.mean(episode_reward)
            self.avg_best_reward = np.mean(episode_best_reward_in_chunk/self.act_steps)
            self.success_rate = np.mean(success_flags)
            self.n_successful_episodes = np.sum(success_flags)
            # Calculate standard deviations
            self.std_episode_reward = np.std(episode_reward)
            self.std_best_reward = np.std(episode_best_reward_in_chunk/self.act_steps)
            self.std_success_rate = np.std(success_flags)
            # Calculate average length of valid episodes and its standard deviation
            episode_lengths = np.array([end - start + 1 for _, start, end in episodes_start_end])*self.act_steps # account for multiple steps
            self.avg_episode_length = np.mean(episode_lengths)
            self.std_episode_length = np.std(episode_lengths)
        else:
            episode_reward = np.array([])
            self.n_episode_finished = 0
            self.n_successful_episodes = 0
            self.avg_episode_reward = 0
            self.std_episode_reward = 0
            self.avg_best_reward = 0
            self.std_best_reward = 0
            self.success_rate = 0
            self.std_success_rate = 0
            self.avg_episode_length = 0
            self.std_episode_length = 0
            self.episode_success_flags = []
            log.info(colored("[WARNING] No episode completed within the iteration!", color="red", on_color="on_red"))

class PPOFlowImgBufferGPU(PPOEvalBufferGPU):
    def __init__(self,
                 device,
                 dtype,
                 n_rollout_steps,
                 n_envs, 
                 n_denoising_steps,
                 n_action_steps, 
                 act_steps,
                 max_action_dim,
                 n_cond_step,
                 visuomotor_obs_dict,
                 language_key,
                 reward_scale_const,
                 reward_scale_running,
                 ignore_nextvalue_when_truncated,
                 success_rew_threshold_in_chunk, 
                 gamma,
                 gae_lambda
                 ):
        """
        Args:
        ---------
            n_rollout_steps: int. The number of rollout steps.
            n_envs: int. The number of parallel environments.
            n_denoising_steps: int. The number of denoising steps.
            n_action_steps: int. The number of action steps.
            act_steps: int. The number of actually deployed action steps.
            max_action_dim: int. The maximum action dimension supported by the model, which could contain paddings on the dimensions that are not used in the robot hardware in deployment. 
            n_cond_step: int. The number of conditioning steps.
            visuomotor_obs_dict: dict. The dictionary of visuomotor observations.
            language_key: str. The key of language instructions.
            reward_scale_const: float. The constant reward scale.
            reward_scale_running: bool. Whether to use running reward scaling.
            ignore_nextvalue_when_truncated: bool. Whether to ignore the next-step value for truncated rollout steps.
            success_rew_threshold_in_chunk: float. The lowest aggregated reward in the chunk that indicates their is a success in action execution. This depends on the chunk size, aggregation method, and reward scale. 
            gamma: float. The reward discount factor.
            gae_lambda: float. The lambda parameter for GAE.
            device: str. The device to store the buffer.
            dtype: torch.dtype. The data type of the buffer.
        """
        # we do not call super().__init__() here because we need to overload too many functions. 
        
        # Device and dtype of the stored data
        self.buffer_device = device
        self.buffer_dtype = dtype
        
        # Number of rollout steps in each iteration and number of parallel environments
        self.n_rollout_steps = n_rollout_steps
        self.n_envs = n_envs
        
        # Number of denoising (integration) steps
        self.n_denoising_steps = n_denoising_steps
        # Model's output action chunk size. 
        self.n_action_steps = n_action_steps
        # Actually deployed action chunk size. We do open-loop control within it. 
        self.act_steps = act_steps
        # Full, padded action dimension that is applicable to various robot hardware structures. This variable is used to initialize the chains_trajs. 
        self.max_action_dim = max_action_dim
        
        # TODO: support multiple camera inputs. 
        self.visuomotor_obs_dim  = visuomotor_obs_dict
        self.language_key = language_key
        if not visuomotor_obs_dict:
            raise ValueError(f"visuomotor_obs_dict is empty, cannot initialize the buffer.")
        if not language_key:
            raise ValueError(f"language_key is empty, cannot initialize the buffer.")
        
        # TODO: support multiple conditioning steps. 
        self.n_cond_step = n_cond_step  # currently not used anywhere. 
        if self.n_cond_step > 1:
            raise NotImplementedError(f"Currently we do not support multiple conditioning steps (n_cond_step={n_cond_step}). Please check your config.")
        
        # Reward (by default, we use per-step reward. the environment wrapper should provide the per-step reward from episode success flags.)
        self.reward_scale_const = reward_scale_const
        self.reward_scale_running =reward_scale_running
        if self.reward_scale_running:
            self.running_reward_scaler = RunningRewardScalerTensor(num_envs=n_envs, gamma=gamma, device = self.buffer_device)
        self.ignore_nextvalue_when_truncated = ignore_nextvalue_when_truncated
        self.success_rew_threshold_in_chunk:float = success_rew_threshold_in_chunk # the lowest aggregated reward in the chunk that indicates their is a success in action execution. 
        # Reward Discount and GAE factor
        self.gamma = gamma
        self.gae_lambda = gae_lambda
    
    # overload
    @torch.no_grad
    def reset(self):
        # Store tensor observations (visual + proprioception)
        self.visuomotor_trajs = {
            k: torch.zeros((self.n_rollout_steps, self.n_envs, *self.visuomotor_obs_dim[k]), 
                          dtype=self.buffer_dtype, device=self.buffer_device)
            for k in self.visuomotor_obs_dim if self.visuomotor_obs_dim[k] is not None  # Skip language key (None shape)
        }
        # Store language instructions separately as list of lists.         # [step][env] -> string instruction
        self.language_trajs = [[None for _ in range(self.n_envs)] for _ in range(self.n_rollout_steps)]
        
        self.reward_trajs = torch.zeros((self.n_rollout_steps, self.n_envs), dtype=self.buffer_dtype, device=self.buffer_device)
        self.terminated_trajs = torch.zeros((self.n_rollout_steps, self.n_envs), dtype=self.buffer_dtype, device=self.buffer_device)
        self.firsts_trajs = torch.zeros((self.n_rollout_steps + 1, self.n_envs), dtype=self.buffer_dtype, device=self.buffer_device)


        # We need to record full-size chains because model.embed_suffix requires the full-size padded actions. 
        # For the other stuffs like value and logprobs we can truncate and normalize them to save storage and compute. 
        self.chains_trajs = torch.zeros((self.n_rollout_steps, self.n_envs, self.n_denoising_steps + 1, self.n_action_steps, self.max_action_dim), dtype=self.buffer_dtype, device=self.buffer_device)
        self.value_trajs = torch.zeros((self.n_rollout_steps, self.n_envs), dtype=self.buffer_dtype, device=self.buffer_device)
        self.logprobs_trajs = torch.zeros((self.n_rollout_steps, self.n_envs), dtype=self.buffer_dtype, device=self.buffer_device)
        # self.logprobs_trajs = torch.zeros((self.n_rollout_steps, self.n_envs, self.n_denoising_steps, self.n_action_steps, self.max_action_dim), dtype=self.buffer_dtype, device=self.buffer_device)
    
    @torch.no_grad
    def make_dataset(self):
        """
        Flatten the dimension on number of parallel environments and rollout steps. Later we will shuffle 
        them into batches for gradient descent. 
        
        obs:
            self.language_key: a list of language instructions (sentences) corresponding to self.n_rollout_steps x self.n_envs. 
        """
        # Construct observation dictionary. 
        # Flatten visuomotor(tensor) observations
        obs = {
            k: self.visuomotor_trajs[k].clone().detach().flatten(0, 1)
            for k in self.visuomotor_trajs
        }
        # Flatten language instructions
        if self.language_key and hasattr(self, 'language_trajs'):
            # Flatten language instructions: [step, env] -> [step * env]
            obs.update({
                self.language_key: 
                    [self.language_trajs[step][env]
                    for step in range(self.n_rollout_steps)  
                    for env in range(self.n_envs)]
                    })
        else:
            raise ValueError(f"language_key is empty, cannot make VLA dataset.")
        
        chains = self.chains_trajs.flatten(0, 1)
        returns = self.returns_trajs.flatten(0, 1)
        values = self.value_trajs.flatten(0, 1)
        advantages = self.advantages_trajs.flatten(0, 1)
        logprobs = self.logprobs_trajs.flatten(0, 1)

        return obs, chains, returns, values, advantages, logprobs
    
    # overload
    @torch.no_grad
    def add(self, step, batch, chains_actions_venv, log_probs_venv, values_venv, reward_venv, terminated_venv, truncated_venv):
        """Add tensors to buffer at the given step.
        
        Args:
            step: Current step index
            batch: Dict of visuomotor tensors and language instructions
                batch:dict[str, Tensor] 
                batch = {
                    self.proprioception_key: proprioception,                           # Tensor[B, state_dim]. B=num_envs. 
                    self.language_key: language_instruction,                           # A length-B list of strings. 
                    self.rgb_keys[0]:  rgb_image_camera_0                              # Images from camera 0: Tensor[B, C, H, W]
                    self.rgb_keys[1]:  rgb_image_camera_1                              # Images from camera 1: Tensor[B, C, H, W]
                    ...
                    self.rgb_keys[num_cameras-1]:  rgb_image_camer_{num_cameras-1}     # Images from the last camera: Tensor[B, C, H, W]
            chains_actions_venv: Action chain tensor
            log_probs_venv: Log probability tensor
            values_venv: Value tensor
            reward_venv: Reward tensor
            terminated_venv: Termination tensor
            truncated_venv: Truncation tensor
        """
         
        # Store tensor observations (visual + proprioception)
        try:
            for k in self.visuomotor_trajs:
                self.visuomotor_trajs[k][step] = batch[k].to(dtype=self.buffer_dtype, device=self.buffer_device)
            # Store language instructions separately
            if self.language_key and self.language_key in batch:
                for env_idx in range(self.n_envs):
                    self.language_trajs[step][env_idx] = batch[self.language_key][env_idx]
        except Exception as e:
            print(f"Meet error when processing prev_obs_venv={batch.keys()}")
            raise e
        # Store action chains and values
        self.chains_trajs[step] = chains_actions_venv.to(dtype=self.buffer_dtype, device=self.buffer_device)
        self.logprobs_trajs[step] = log_probs_venv.to(dtype=self.buffer_dtype, device=self.buffer_device)
        self.value_trajs[step] = values_venv.to(dtype=self.buffer_dtype, device=self.buffer_device)
        self.reward_trajs[step] = reward_venv.to(dtype=self.buffer_dtype, device=self.buffer_device)
        self.terminated_trajs[step] = terminated_venv.to(dtype=self.buffer_dtype, device=self.buffer_device)
        self.firsts_trajs[step + 1] = (terminated_venv | truncated_venv).to(dtype=self.buffer_dtype, device=self.buffer_device)  # done_venv
    
    
    @torch.no_grad
    def update(self, nextobs_venv:dict, runner: PI0PolicyPPO):
        """
        Normalized reward, update advantages and returns. 
        The input arguments are used to update the value function V(s_t+1) for the truncated actions in the buffer. 
        """
        # normalize reward with running variance
        self.normalize_reward()
        
        self.update_adv_returns(nextobs_venv, runner)
    
    @torch.no_grad
    def update_adv_returns(self, nextobs_batch:dict, runner: PI0PolicyPPO):
        """
        This function is used to update the advantages and returns for the truncated actions in the buffer. 
        The input arguments are used to update the value function V(s_t+1) for the truncated actions in the buffer. 
        The runner should have attribute  `get_value`.  
        """
        # Prepare observation dictionary for critic. Here we do not move the tensors to the model's device, because the fetch_obs_from_env function already moved them to the model's device. 
        nextobs_batch_ts = {
            key: nextobs_batch[key] for key in self.visuomotor_obs_dim
        }
        nextobs_batch_ts.update({self.language_key: nextobs_batch[self.language_key]})
        
        # Compute adv with GAE
        self.advantages_trajs = torch.zeros(self.n_rollout_steps, self.n_envs, device=self.buffer_device)
        lastgaelam = 0
        for t in reversed(range(self.n_rollout_steps)):
            # get V(s_t+1)
            if t == self.n_rollout_steps - 1:
                nextvalues = runner.get_value(batch=nextobs_batch_ts, ignore_value=self.ignore_nextvalue_when_truncated) # returns float 0.0 if ignore_nextvalue_when_truncated is True, else returns a tensor. 
                if isinstance(nextvalues, torch.Tensor):
                    nextvalues = nextvalues.to(device=self.buffer_device, dtype=self.buffer_dtype)
            else:
                nextvalues = self.value_trajs[t + 1]
            # delta = r + gamma*V(st+1) - V(st)
            non_terminal = 1.0 - self.terminated_trajs[t]
            delta = (
                self.reward_trajs[t] * self.reward_scale_const
                + self.gamma * nextvalues * non_terminal
                - self.value_trajs[t]
            )
            # A = delta_t + gamma*lamdba*delta_{t+1} + ...
            self.advantages_trajs[t] = lastgaelam = (
                delta
                + self.gamma * self.gae_lambda * non_terminal * lastgaelam
            )
        # compute return
        self.returns_trajs = self.advantages_trajs + self.value_trajs

    @torch.no_grad
    def normalize_reward(self):
        """
        normalize self.reward_trajs
        """
        if self.reward_scale_running:
            reward_trajs_transpose = self.running_reward_scaler(reward=self.reward_trajs.T, first=self.firsts_trajs[:-1].T)
            self.reward_trajs = reward_trajs_transpose.T
    
    @torch.no_grad
    def get_explained_var(self, values, returns):
        """
        Get the explained variance of the value function. 
        """
        # Assuming values and returns are already tensors
        y_pred = values.detach()  # Detach to prevent gradient tracking
        y_true = returns.detach()
        var_y = y_true.var().item()
        explained_var = (float('nan') if var_y == 0 else 1 - ((y_true - y_pred).var().item() / var_y))
        return explained_var  # Returns a floating point number
    
    