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


#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Revised from Kang Chen lerobot/common/policies/pi0/modeling_pi0.py
https://github.com/chenkang455/PI0_RL
"""

"""
π0: A Vision-Language-Action Flow Model for General Robot Control

[Paper](https://www.physicalintelligence.company/download/pi0.pdf)
[Jax code](https://github.com/Physical-Intelligence/openpi)

Designed by Physical Intelligence. Ported from Jax by Hugging Face.

Install pi0 extra dependencies:
```bash
pip install -e ".[pi0]"
```

Example of finetuning the pi0 pretrained model (`pi0_base` in `openpi`):
```bash
python lerobot/scripts/train.py \
--policy.path=lerobot/pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of finetuning the pi0 neural network with PaliGemma and expert Gemma
pretrained with VLM default parameters before pi0 finetuning:
```bash
python lerobot/scripts/train.py \
--policy.type=pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of using the pi0 pretrained model outside LeRobot training framework:
```python
policy = Pi0Policy.from_pretrained("lerobot/pi0")
```

"""

import torch
from torch import Tensor, nn
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.pi0.pi0_helpers import *
from lerobot.common.policies.pi0.modeling_pi0 import PI0FlowMatching

####################
# for rl functions
from typing import Literal
from collections import namedtuple
from torch.distributions import Normal
Sample = namedtuple("Sample", "final_actions chains log_probs value")
from custom_model.explore_noise_net import ExploreNoiseNet
from custom_model.mlp import MLP
from typing import Tuple
import logging 
logger = logging.getLogger(__name__)


class PI0PolicyRL(PI0Policy):
    """
    Inherited from LeRobot's PI0Policy class, but instantiates the self.model as PI0FlowMatchingRL that includes value function, noise injection, SDE integration, and other RL functions needed. 
    Reference:
        π0: A Vision-Language-Action Flow Model for General Robot Control [Paper](https://www.physicalintelligence.company/download/pi0.pdf) [Jax code](https://github.com/Physical-Intelligence/openpi)
    
    Note: Action queue management is handled externally by ActionQueueManager, so this class
    focuses purely on RL-specific inference and training functionality.
    """
    def __init__(self, config: PI0Config, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        super().__init__(config, dataset_stats)
    
    # overload. will be called in the super().__init__()
    def build_model(self):
        self.model = PI0FlowMatchingRL(self.config)


class PI0FlowMatchingRL(PI0FlowMatching):
    """
    π0: A Vision-Language-Action Flow Model for General Robot Control

    [Paper](https://www.physicalintelligence.company/download/pi0.pdf)
    [Jax code](https://github.com/Physical-Intelligence/openpi)

    Designed by Physical Intelligence. Ported from Jax by Hugging Face.
    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │Gemma │        │
    │  ┌──────────►│Expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x 10  │        │
    │ │         │  └▲──▲──┘        │
    │ │PaliGemma│   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘
    """

    def __init__(self, config):
        super().__init__(config)

        logger.info(f"\nInitializing {self.__class__.__name__} with RL config.\n")
        # RL functions
        # The additional self.config entries will be appended to the PI0Config before training starts. 
        # Details: 
        # add the value network. 
        self.adv_method=self.config.adv_method
        if self.adv_method == "gae":
            """
            Critic is a mapping from self.config.proj_width to 1. 
            """
            # if no critic is specified, use a linear layer. Otherwise, use an MLP. 
            if self.config.critic is None:
                self.critic = nn.Linear(self.config.proj_width, 1)
            else:
                self.critic = MLP([self.config.proj_width, *self.config.critic.hidden_dims, 1],
                                  append_dim=0,
                                  append_layers=None,
                                  activation_type=self.config.critic.activation_type,
                                  out_activation_type="identity",
                                  use_layernorm=self.config.critic.use_layernorm,
                                  use_layernorm_final=self.config.critic.use_layernorm_final,
                                  dropout=self.config.critic.dropout,
                                  use_drop_final=self.config.critic.use_drop_final,
                                  out_bias_init=self.config.critic.out_bias_init)
            logger.info(f"Critic created: {self.critic}")
        elif self.adv_method == "rloo":
            raise NotImplementedError("RLOO is not implemented yet.")
        elif self.adv_method == "base":
            raise NotImplementedError("Base is not implemented yet.")
        elif self.adv_method == "mean":
            raise NotImplementedError("Mean is not implemented yet.")
        elif self.adv_method == "grpo":
            raise NotImplementedError("GRPO is not implemented yet.")
        else:
            raise ValueError(f"self.adv_method={self.adv_method} is not supported, please check your config.yaml")
        
        # add the noise prediction network (optionally) which receives suffix_out decoded from paligemma_with_expert
        self.sde_mode=self.config.sde_mode
        if self.sde_mode=='reinflow':
            # cache the minimum and maximum log variance of the injected noise
            self.reinflow_noise_logvar_min: float  = self.config.explore_noise_net.noise_logvar_min
            self.reinflow_noise_logvar_max: float  = self.config.explore_noise_net.noise_logvar_max
            if not self.reinflow_noise_logvar_min < self.reinflow_noise_logvar_max:
                raise ValueError(f"reinflow_noise_logvar_min={self.reinflow_noise_logvar_min} must be less than reinflow_noise_logvar_max={self.reinflow_noise_logvar_max}")
            self.reinflow_explore_noise_net = ExploreNoiseNet(in_dim=self.config.proj_width, 
                                                              out_dim=self.config.max_action_dim,
                                                              hidden_dims=self.config.explore_noise_net.hidden_dims,
                                                              activation_type=self.config.explore_noise_net.activation_type,
                                                              noise_logvar_range=[self.reinflow_noise_logvar_min, self.reinflow_noise_logvar_max],
                                                              noise_scheduler_type=self.config.noise_scheduler_type)            
        elif self.sde_mode == 'flow-grpo':
            self.noise_level = self.config.noise_level  # the coefficient a in flow-grpo. 
        else: 
            raise ValueError(f"sde_mode={self.sde_mode} is not supported, methods involving SDE will not function. Please check your config.yaml")
        
        # add the effective action dimension
        if self.config.action_feature is not None:
            self.effective_action_dim = self.config.action_feature.shape[0]  
        else:
            self.effective_action_dim = self.config.max_action_dim
            logger.warning(f"No action feature specified, using max_action_dim={self.effective_action_dim}")
        
        # for clipping denoising actions. 
        self.denoise_action_min = self.config.denoise_action_min
        self.denoise_action_max = self.config.denoise_action_max
        
        # for normalizing the log probability over denoising horizon and action chunk full dimension. 
        self.normalize_denoising_horizon = self.config.normalize_denoising_horizon
        self.normalize_action_chunk_full_dim = self.config.normalize_action_chunk_full_dim
        if self.normalize_action_chunk_full_dim is None:
            raise ValueError(f"normalize_action_chunk_full_dim is not specified in the config. Please check your config.yaml")
        
        # Numerical integration discretization method
        self.time_discretization = self.config.time_discretization
        time_discretization_options=['uniform']
        if not self.time_discretization in time_discretization_options:
            raise NotImplementedError(f"self.time_discretization={self.time_discretization} is unsupported. Please choose one from {time_discretization_options}.")
        
        # Critic preprocessing:
        self.detach_feature_before_input_critic = self.config.detach_feature_before_input_critic
        self.average_critic_input_by_deployed_chunk = self.config.average_critic_input_by_deployed_chunk
        
    ################################################################################################
    # rl functions
    
    def sample_noise_logprob(self, shape, device)->Tuple[Tensor, Tensor]:
        """
        Sample a unit Gaussian noise and optinonally return its log probability. 
        """
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        log_prob = self.get_unit_gaussian_noise_log_prob(noise)
        return noise, log_prob
    
    def get_unit_gaussian_noise_log_prob(self, noise)->Tensor:
        """
        Get the log probability of a unit Gaussian noise. 
        """
        return Normal(0.0, 1.0).log_prob(noise)
    
    def get_unit_gaussian_noise_entropy(self)->Tensor:
        """
        Get the entropy of a unit Gaussian noise. 
        """
        return Normal(0.0, 1.0).entropy()
    
    def get_timesteps(self,num_steps,device)->Tensor:
        """
        Derive the timesteps of the diffusion model (flow, but following the timestep convention of diffusions)
        t is the timestep, which ranges from 1, 1-△t, 1-2△t, ..., 1/K, 0, where △t=1/K under uniform time discretization. 
        """
        if self.time_discretization == 'uniform':
            timesteps = torch.linspace(1, 1 / num_steps, num_steps, device=device)
            timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        else:
            raise NotImplementedError(f"self.time_discretization={self.time_discretization} is unsupported. ")
        return timesteps
    
    def get_flowgrpo_noise_level(self, timesteps:torch.Tensor)->Tensor:
        """
        Here we derive the SDE noise levels from Flow-GRPO paper. https://arxiv.org/abs/2505.05470
        
        \sigma_t = a \sqrt{\frac{t}{1-t}}
        Here, 
        a is the noise_level
        t is the timestep, which ranges from 1, 1-△t, 1-2△t, ..., 1/K, 0, where △t=1/K. 
        to avoid division by zero at sigma_1, we approximate the sigma value with the nearest timestep 1-△t.
        when t=0, there is no noise added and we do not need to compute the sigma value.
        """
        sigmas = self.noise_level * torch.sqrt(
            timesteps / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
        )#         to avoid division by zero at sigma_1, we approximate the sigma value with the nearest timestep 1-△t.
        sigmas = sigmas[:-1] #        when t=0, there is no noise added and we do not need to compute the sigma value.
        return sigmas
    
    def sample_action_chain_logprob_value(
        self, images, img_masks, lang_tokens, lang_masks, state, 
        sample_mode: Literal["ode","sde"] = "ode",
        n_denoise_steps: int = 10,
        clip_intermediate_actions: bool = False,
        get_chains: bool = True,
        get_logprob: bool = False,
        get_value: bool = False
        ) -> Tuple[Tensor, Tensor|None, Tensor|None, Tensor|None]:
        """Description: Sample a chain of latent actions of length `denoise_steps` by solving the ODE/SDE, and compute the log probabilities of the joint distribution. 
        We also compute the values of each latent actions. 
        Args:
            sample_mode: "ode" or "sde"
            n_denoise_steps: the number of denoising steps for ODE/SDE integration. we use this parameter to compute the time discretization scheme. 
            clip_intermediate_actions: whether to clip the intermediate actions to the range [self.denoise_action_min, self.denoise_action_max]
            get_chains: whether to compute the chains of latent actions. 
            get_logprob: whether to compute the log probabilities of the joint distribution
            get_value: whether to compute the value functions for each denoised action in the batches. 
                the value here is w.r.t. observation AND all denoised actions. 
        Returns:
                `final_actions`: torch.Tensor[batch_size, self.config.n_action_steps, self.config.max_action_dim] 
                `chains`: torch.Tensor[batch_size,denoise_steps + 1,self.confign_action_steps,self.config.max_action_dim] 
                `joint_log_prob`: torch.Tensor[batch_size] the joint log probability (normalized) of this denoising chain. 
                `value`: torch.Tensor[batch_size]
        """
        bsize = state.shape[0]
        device = state.device
        actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
        # Prepare visua-language prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Cache image and language representations
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids, # type: ignore
            past_key_values=None,
            inputs_embeds=[prefix_embs, None], # type: ignore
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        
        # Optionally pre-allocate output tensors to save time. 
        chains = torch.zeros(bsize, n_denoise_steps + 1, self.config.n_action_steps, self.config.max_action_dim, device=device)
        if get_logprob:
            log_prob_chain = torch.zeros(bsize, n_denoise_steps+1, self.config.n_action_steps, self.config.max_action_dim, device=device)
        if get_value:
            value_chain = torch.zeros(bsize, n_denoise_steps, device=device)
        
        # Sample initial state
        if get_logprob:
            x_t, log_prob_t = self.sample_noise_logprob(actions_shape, device)
            log_prob_chain[:, 0] = log_prob_t
        else:
            x_t = self.sample_noise(actions_shape, device)
        if get_chains:
            chains[:, 0] = x_t  # Store initial state  # type: ignore

        # Denoising loop
        for denoise_step in range(n_denoise_steps):
            x_t_mean, x_t_std, value_t = self.get_mean_std_val_at_denoise_step(
                sample_mode,
                clip_intermediate_actions, 
                n_denoise_steps,
                denoise_step,
                x_t, # type: ignore
                state,
                prefix_pad_masks, # visual-language prefix padding masks
                past_key_values   # visual-language prefix representations
            )
            # When doing SDE inference, sample x_t \sim \mathcal{N}(·|x_t_mean, x_t_std^2) with re-parametrization trick. If doing ODE inference, x_t_std is 0 so it is still fine. 
            unit_noise=self.sample_noise(x_t.shape, device) # type: ignore
            x_t = x_t_mean + x_t_std * unit_noise
            
            # Store results directly in pre-allocated tensors
            if get_chains:
                chains[:, denoise_step + 1] = x_t # type: ignore
            if get_logprob:
                log_prob_t = get_gaussian_log_prob(x_t, x_t_mean, x_t_std)
                log_prob_chain[:, denoise_step + 1] = log_prob_t # type: ignore
            if get_value:
                value_chain[:, denoise_step] = value_t # type: ignore
        
        # Average and normalize the joint log probability over the denoising steps. 
        if get_logprob:
            # we only record the joint log probability over the denoising chain's effective action dimensions to save compute and storage. 
            joint_log_prob = log_prob_chain[:,:,:self.config.act_steps,:self.effective_action_dim].sum(dim=(-3,-2,-1)) # sum over denoising steps and effective action chunk (open-loop assume independent). [B, Ta, Da] --> [B]
            if self.normalize_denoising_horizon:
                joint_log_prob = joint_log_prob / (1+n_denoise_steps)
            if self.normalize_action_chunk_full_dim:
                joint_log_prob = joint_log_prob / (self.config.act_steps * self.effective_action_dim)
        else:
            joint_log_prob = None
        
        # Average value estimate over denoising steps to obtain the value for the inference process.  
        if get_value:
            value=value_chain.mean(dim=1)
        else:
            value=None
        
        return x_t, chains, joint_log_prob, value

    def get_mean_std_val_at_denoise_step(self,
                                        sample_mode: str,
                                        clip_intermediate_actions: bool,
                                        n_denoise_steps: int,
                                        denoise_step_id: int | torch.LongTensor,
                                        x_t: Tensor,
                                        state: Tensor,
                                        prefix_pad_masks,
                                        past_key_values)->Tuple[Tensor,Tensor,Tensor]:
        """Compute the mean, std and value for the action at denoising step `idx`
        Args:
            sample_mode: "ode" or "sde"
            clip_intermediate_actions: whether to clip the intermediate actions to the range [self.denoise_action_min, self.denoise_action_max]
            denoise_steps: the number of denoising steps. we use this parameter to compute the time discretization scheme. 
            idx: long tensor or scalar, the index of the current denoising step. this will affect the noise level of the SDE with flow-grpo sampling. 
            x_t: full-shape (padded) denoised actions at different timesteps packed into a batch. torch.Tensor[B, self.config.n_action_steps, self.config.max_action_dim]
            state: the current state
            prefix_pad_masks: the padding masks of the prefix
            past_key_values: the past key values of the paligemma model
        
        Returns: 
        Let t is the denoise step. 
        `x_t_mean, x_t_std, value_t`: torch.Tensor[B, self.config.n_action_steps, self.config.max_action_dim]. 
        the mean, variance and value for each sample in the input batch, which has various denoising steps. 
        `value_t`: conditions on the full observation (visual-language-state) and the current denoising action (o, a_t, t). 
        
        Eventually we sample x_t = x_t_mean + x_t_std * unit_noise
        """
        # parameters 
        bsize = state.shape[0]
        device = state.device
        timesteps= self.get_timesteps(n_denoise_steps,device)
        # delta is negative following the convention of physical intelligence's pi-zero codebase. (though it is the opposite of what we use in flow matching models.)
        delta = timesteps[denoise_step_id + 1] - timesteps[denoise_step_id]
        abs_delta = -delta
        # t_i is the current denoising time
        t_i = timesteps[denoise_step_id]
        # velocity prediction based on the cached prefix. 
        suffix_out = self.get_suffix_out(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            t_i.expand(bsize),
        ) # [batchsize, self.config.n_action_steps, self.config.proj_width]
        v_t = self.action_out_proj(suffix_out)                                                                 # velocities of [batchsize, self.config.n_action_steps, self.config.max_action_dim]
        
        # value prediction for each denoise step, grouped in batches. 
        if self.adv_method == "gae":
            # average over
            if self.average_critic_input_by_deployed_chunk:
                critic_input_feature = torch.mean(suffix_out[:,:self.config.act_steps,:],dim = 1,keepdim=False) # average over the actually deployed action chunk. 
            else:
                critic_input_feature = torch.mean(suffix_out,dim = 1,keepdim=False)                             # average over the full output action chunk. 
            if self.detach_feature_before_input_critic:
                critic_input_feature = critic_input_feature.detach()                                            # stop critic gradient from flowing back to the policy. make critic an auxiliary network. 
            value_t = self.critic(critic_input_feature)[:,0]                                                    # the critic net outputs [batchsize, 1] and we squeeze the last dimension to make it value_t stay in the shape of [batchsize]
        else:
            value_t = torch.zeros((bsize),device=device)
        
        # sampling x_t from N(x_t_mean, x_t_std^2)
        if sample_mode == "ode":
            """
            Flow-ODE inference:
                x_{t+\Delta t} = x_t + v_t \Delta t
            """
            weight_x = torch.ones_like(x_t)
            weight_v = torch.ones_like(v_t)
            weight_std = torch.zeros_like(x_t) # warning: zero noise may cause numerical instability when calculating the log probability. do not use this option during RLFT, only do that in eval mode. 
            sigma_i  =   torch.zeros_like(x_t) 
            if delta.dim() == 0:
                delta = delta * torch.ones_like(x_t)
            elif delta.dim() == 1:
                delta = delta.unsqueeze(-1).unsqueeze(-1).expand_as(x_t)
            else:
                raise ValueError(f"Unexpected delta dimensions: {delta.dim()}")
        elif sample_mode == "sde":
            if self.sde_mode=='flow-grpo':
                """
                Flow-GRPO: ODE-SDE conversion.  
                    x_{t+\Delta t}=x_t+\left[v_\theta(x_t, t)+\frac{\sigma_t^2}{2 t}(x_t+(1-t) v_\theta(x_t, t))\right] \Delta t+\sigma_t \sqrt{\Delta t} \epsilon
                Reference: Eq.12 in https://arxiv.org/abs/2505.05470 
                """
                sigmas=self.get_flowgrpo_noise_level(timesteps)
                sigma_i = sigmas[denoise_step_id]
                if isinstance(denoise_step_id,int):
                    weight_x = 1 + sigma_i**2 / (2 * t_i) * delta
                    weight_v = 1 + sigma_i**2 / (2 * t_i) * (1 - t_i) 
                    weight_std = torch.sqrt(abs_delta)
                else:
                    weight_x = torch.ones_like(sigma_i) + sigma_i**2 / (2 * t_i) * delta
                    weight_x = weight_x[:,None,None].expand_as(x_t)
                    
                    weight_v = torch.ones_like(sigma_i) + sigma_i**2 * (1 - t_i) / (2 * t_i)                    
                    weight_v = weight_v[:,None,None].expand_as(x_t)
                    
                    weight_std = torch.sqrt(abs_delta)
                    weight_std = weight_std[:,None,None].expand_as(x_t)
                    
                    sigma_i = sigma_i[:,None,None].expand_as(x_t)
                    delta = delta[:,None,None].expand_as(x_t)
            elif self.sde_mode == 'reinflow':
                """
                    ReinFlow integration. 
                        x_{t+\Delta t} = x_t + v_t \Delta t + \sigma_t(o, x_t, t) \epsilon
                """
                weight_x=torch.ones_like(x_t)
                weight_v=torch.ones_like(v_t)
                weight_std=torch.ones_like(x_t)                                 #* torch.sqrt(abs_delta)
                sigma_i = self.reinflow_explore_noise_net.forward(suffix_out) 
                if delta.dim() == 0:
                    delta = delta * torch.ones_like(x_t)
                elif delta.dim() == 1:
                    delta = delta.unsqueeze(-1).unsqueeze(-1).expand_as(x_t)
                else:
                    raise ValueError(f"Unexpected delta dimensions: {delta.dim()}")
            else:
                raise NotImplementedError(f"Unsupported sde_mode={self.sde_mode}")
            
        else:
            raise NotImplementedError(f"Unsupported sample_mode={sample_mode}")
        
        # the distribution from which to draw samples. 
        x_t_mean = weight_x *  x_t  + weight_v * v_t  * delta    # delta is negative following the convention of physical intelligence's pi-zero codebase. (though it is the opposite of what we use in flow matching models.)
        if clip_intermediate_actions:  # prevent excessively large latent actions wander into OOD points of the model. 
            x_t_mean = x_t_mean.clamp(self.denoise_action_min, self.denoise_action_max)
        
        # the standard deviation of the noise. 
        x_t_std = weight_std * sigma_i
        return x_t_mean, x_t_std, value_t

