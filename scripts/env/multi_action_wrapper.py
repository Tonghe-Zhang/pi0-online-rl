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
import gymnasium
from mani_skill.envs.sapien_env import BaseEnv

class MultiActionWrapper(gymnasium.Wrapper):
    def __init__(self, 
                 env:BaseEnv, 
                 reward_agg_method="sum"):   # never use other types. 
        """
        This wrapper is used to wrap GPU parallelized environments to support action-chunk open-loop control. 
        **Please wrap this around a maniskill BaseEnv, and put the ManiSkillVectorEnv outside this wrapper, as 
        this wrapper currently does not support partial reset (while the ManiSkillVectorEnv does). **
        It is used to control the environment in a batch of environments, and the action is a sequence of actions. 
        The reward is aggregated until the first termination or truncation is met. 
        The success is the logical or of the success of all the actions in the chunk. 
        The terminated and truncated are the logical or of the terminated and truncated of all the actions in the chunk. 
        
        Args:
            env: the environment to wrap. 
            reward_agg_method: the method to aggregate the reward. currently only support "sum". 
            
        Returns:
            MultiActionWrapper: the wrapped environment.        
        """
        super().__init__(env)
        self.env: BaseEnv = env
        self.num_envs = env.unwrapped.num_envs # type: ignore
        self.reward_agg_method=reward_agg_method
        if self.reward_agg_method not in ["sum", "mean", "max", "min"]:
            raise ValueError(f"Invalid reward aggregation method: {self.reward_agg_method}")
        #################################################
        self.debug_mode=False #True  # for development only.  
        
    def reset(self, seed=None, options: dict = {}):
        obs, info = self.env.reset(seed=seed, options=options) # type: ignore
        return obs, info

    def step(self, action):
        """
        open-loop control within the action chunk.
        
        Args:
            action: (num_envs, act_steps, action_dim)
        Returns:
            _next_obs: (num_envs, obs_shape)    each env's next observation received after executing the last action in the chunk. 
            aggregated_reward: (num_envs)       each env's aggregated reward until the first termination or truncation is met. 
            terminated_in_chunk: (num_envs)     whether this env met a single termination in this action chunk. 
            truncated_in_chunk: (num_envs)      whether this env met a single truncation in this action chunk. 
            _info: (num_envs)                    each env's info after executing the last action in the chunk, with the 'success' flag overwritten as whether there is a success in all the actions in this chunk. 
    
        TODO: support 'fail' flag. 
        """
        
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"Action must be a torch.Tensor, but got {type(action)}")
        if action.ndim != 3:
            raise ValueError(f"Action must be a 3D tensor of shape [num_envs, act_steps, action_dim] but got {action.ndim}D tensor")
        if action.shape[0] != self.num_envs:
            raise ValueError(f"Action must have {self.num_envs} environments, but got {action.shape[0]}")
        
        sim_device=action.device
        sim_dtype=action.dtype
        n_act_steps = action.shape[-2]
        
        self.reward=torch.zeros(self.num_envs, n_act_steps, device=sim_device, dtype=sim_dtype)
        # In case reward_mode is None, record whether there is success within this action chunk.  
        self.success = torch.zeros(self.num_envs, device=sim_device, dtype=sim_dtype).bool()  
        
        self.terminated=torch.zeros(self.num_envs, n_act_steps, device=sim_device, dtype=sim_dtype).bool()
        self.truncated=torch.zeros(self.num_envs, n_act_steps, device=sim_device, dtype=sim_dtype).bool()
        
        
        _terminated = False
        _truncated  = False
        
        
        for act_step in range(n_act_steps):
            act = action[:, act_step]
            # open-loop control within the action chunk for all environments
            _next_obs, _reward, _terminated, _truncated, _info = self.env.step(act)
            # record terminated and truncate info for each env. 
            self.terminated[:,act_step]=_terminated
            self.truncated[:,act_step]=_truncated
            # when not done, record reward
            self.reward[:,act_step]=_reward # [num_envs]
            # as long as there is a single step that is successful, we record this episode as succcessful. 
            if 'success' in _info.keys():
                self.success = self.success | _info['success']
            else: # for debug use. current envs all have success flags. 
                raise ValueError(f"** 'success' not in _info.keys()! _info={_info.keys()}")
        
        # Now I want you to summarize what happened in this action chunk, 
        # When we calculate the terminated/truncated for each envs, we flag terminated_in_chunk[env_id] = True if there is a single terminated in that row, and
        # we do the same for the truncated. 
        terminated_in_chunk = self.terminated.any(dim=1)  # [num_envs]
        truncated_in_chunk = self.truncated.any(dim=1)    # [num_envs]
        
        # When we calculate the reward for each environment, we record the sum of reward at each action step until the first termination or truncation is met. this is 
        # executed for all envs in parallel and store the result in aggregated_reward which is of shape [num_envs]. 
        done_mask = (self.terminated | self.truncated)  # [num_envs, n_act_steps]
        # Find the first done index for each env, or n_act_steps if never done
        first_done_idx = torch.where(
            done_mask.any(dim=1),
            done_mask.float().argmax(dim=1),
            torch.full((self.num_envs,), n_act_steps-1, device=done_mask.device)
        )  # [num_envs]
        # Create a mask for each env: True for steps to include in sum
        step_range = torch.arange(n_act_steps, device=done_mask.device).unsqueeze(0)  # [1, n_act_steps]
        # For each env, mask steps up to and including first_done_idx (or all if never done)
        mask = step_range <= first_done_idx.unsqueeze(1)  # [num_envs, n_act_steps]
        # Aggregate rewards as the sum of the reward in the action chunk using the mask
        if self.reward_agg_method == "sum":
            aggregated_reward = (self.reward * mask).sum(dim=1)  # [num_envs]
        else:
            raise NotImplementedError(f"** {self.reward_agg_method} is not implemented!")

        # Overwrite last-step success with action chunk success. 
        _info['success']=self.success # as long as there is a single step that is successful, we record this chunk as succcessful. 
        
        #################################################
        if self.debug_mode:
            import logging 
            logger=logging.getLogger(__name__)
            logger.info(f"DEBUG::MultiActionWrapper: terminated_in_chunk={terminated_in_chunk}, truncated_in_chunk={truncated_in_chunk}, _info['success']={_info['success']}, aggregated_reward={aggregated_reward}")
        #################################################
        
        return _next_obs, aggregated_reward, terminated_in_chunk, truncated_in_chunk, _info
    
    
    def close(self):
        return self.env.close()
    
    def _aggregate(self, agg_method:str, value_deque, sim_device:torch.device):
        if agg_method == "sum":
            aggregated_value = torch.sum(torch.stack(value_deque), dim=0).to(sim_device)
        elif agg_method == "mean":
            aggregated_value = torch.mean(torch.stack(value_deque), dim=0).to(sim_device)
        elif agg_method == "max":
            aggregated_value = torch.max(torch.stack(value_deque), dim=0).values.to(sim_device)
        elif agg_method == "min":
            aggregated_value = torch.min(torch.stack(value_deque), dim=0).values.to(sim_device)
        else:
            raise ValueError(f"Invalid aggregation method: {agg_method}")
        return aggregated_value
    