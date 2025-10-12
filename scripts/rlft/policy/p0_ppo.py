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


from typing import Literal, Tuple
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.distributions import Normal
from lerobot.common.policies.pi0.modeling_pi0_rl import PI0PolicyRL
from lerobot.common.policies.pi0.pi0_helpers import make_att_2d_masks, get_gaussian_log_prob
import logging
logger = logging.getLogger(__name__)

class PI0PolicyPPO(PI0PolicyRL):
    def __init__(
        self,
        config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None
        ):
        super().__init__(config, dataset_stats)
        # PPO ratio clipping
        self.clip_ppo_eps = config.__dict__.get("ppo_clipping_eps", 0.01)
        # Value function clipping. None means no clipping. 
        self.clip_vloss_coef = config.__dict__.get("clip_vloss_coef", None) 
        # prevent policy collapse. 
        self.logprob_min = config.__dict__.get("logprob_min", -3)  
        self.logprob_max = config.__dict__.get("logprob_max", 3)
        # Denoised action clipping. 
        self.clip_intermediate_actions = config.__dict__.get("clip_intermediate_actions", False)
        # Number of denoising steps for evaluation, it can be different from training. 
        self.num_steps_eval = config.__dict__.get("num_steps_eval", self.config.num_steps)
        # When using reinflow, we can adjust the bounds of the injected noise with the noise scheduler. 
        self.noise_scheduler_type:str=config.__dict__.get("noise_scheduler_type", "const")

    def _prepare_visual_language_state_input(self,batch):
        batch = self.normalize_inputs(batch)
        # prepare
        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        state = self.prepare_state(batch)
        return images, img_masks, lang_tokens, lang_masks, state
    
    @torch.no_grad
    def _run_full_inference(self, batch: dict[str, Tensor], sample_mode: Literal["ode", "sde"], denoise_steps: int, clip_intermediate_actions=False, get_chains=True, get_logprob=True, get_value=True):
        """Full inference by integrating the SDE/ODE of noise-injected flow matching model. 
        Truncate the model's output action chunk to the effective shape and do unnormalization, so that actions are ready to be deployed. 
        
        **Note on slicing the output**:
            We slice the final denoised actions, pick only the first `self.config.act_steps` in the model't output action chunk and the first 
            `self.model.effective_action_dim` dimensions to remove the padding and get the real actions to be sent to the simulator (real world). 
            The log probabilities and value functions are also sliced in the same way, because they will be used to compute the PPO loss based on the 
            actual actions collected from the environment. 
            
            However, the chains are not sliced, because they will be used to derive the log probs from the model, which expects a full, although redundant, output. 

        Args:
            batch: dict[str, Tensor]
            sample_mode: str, "ode" or "sde"
            denoise_steps: int. It indicates the integration steps of the SDE/ODE. Note that we can choose different denois_steps for eval or training. This variable is used to derive the time-discretization scheme. 
            clip_intermediate_actions: bool. Whether to clip the intermediate actions to the range [self.denoise_action_min, self.denoise_action_max]
            get_logprob: bool. Whether to return the log probabilities of the actions.
            get_value: bool. Whether to return the value functions of the actions.
        
        Returns:
            tuple of (deployable_action, chains, log_probs, values). 
            Here, `deployable_action` is the action with the correct and often fewer dimension that can be deployed to the environment, 
            and it is never directly recorded in the buffer. 
            while other quantities will be recorded in the buffer during training, and they match the full output shape of the model. 
            `deployable_action_chunk`: torch.Tensor[batch_size, self.config.act_steps, self.model.effective_action_dim]
            `chains`: torch.Tensor[batch_size,denoise_steps + 1,self.config.n_action_steps,self.config.max_action_dim] 
            `log_probs`: torch.Tensor[batch_size,denoise_steps,self.config.n_action_steps,self.config.max_action_dim] 
            `values`: torch.Tensor[batch_size]
        """
        # prepare input
        images, img_masks, lang_tokens, lang_masks, state= self._prepare_visual_language_state_input(batch)
        # sample
        final_actions, chains, log_probs, values = \
            self.model.sample_action_chain_logprob_value(images=images, img_masks=img_masks, lang_tokens=lang_tokens, lang_masks=lang_masks, state=state, 
                                                      sample_mode=sample_mode, n_denoise_steps=denoise_steps, clip_intermediate_actions=clip_intermediate_actions, 
                                                      get_chains=get_chains, get_logprob=get_logprob, get_value=get_value)
            
        # Truncate model's output actions to effective shape. 
        # We only select the first `self.config.act_steps` actions in the model's output action chunk during execution. Those redundant padding action dimensions will also be removed. 
        deployable_action_chunk = final_actions[:, :self.config.act_steps, :self.model.effective_action_dim]
        # Unnormalize the actions to the original action space, and this action will be directly deployed (in the simulator or in real world, as a command to the low-level controller)
        deployable_action_chunk = self.unnormalize_outputs({"action": deployable_action_chunk})["action"]
        if self.config.adapt_to_pi_aloha:
            deployable_action_chunk = self._pi_aloha_encode_actions(deployable_action_chunk)
        
        # Optionally return log probs and values
        if get_logprob and log_probs==None:
            raise ValueError(f"log_probs==None when you use get_logprob. Please check relevant logic. ")
        if get_value and values==None:
            raise ValueError(f"values==None when you use get_logprob. Please check relevant logic. ")
        
        return deployable_action_chunk, chains, log_probs, values
    
    @torch.no_grad
    def select_action(self, batch: dict, evaluate: bool = False)->Tuple[Tensor, Tensor|None, Tensor|None, Tensor|None]:
        """Select the action to be sent to the simulator (real world) when evaluating the policy; 
        also record the chains, log probabilities and value functions for each denoising step when training with RLFT, in which the output will be stored in the buffer. 
        This function does not record the gradients. 
        
        Args:
            batch: dict[str, Tensor]
            evaluate: bool. Whether to evaluate the policy.
        
        Returns:
        
            `deployable_action_chunk`: torch.Tensor[batch_size, self.config.act_steps, self.model.effective_action_dim]
            
            `chains`: torch.Tensor[batch_size,denoise_steps + 1,self.config.n_action_steps,self.config.max_action_dim] or None
            
            `log_probs`: torch.Tensor[batch_size,denoise_steps,self.config.n_action_steps,self.config.max_action_dim]  or None
            
            `values`: torch.Tensor[batch_size] or None
        """
        if not evaluate:
            """
            This option is for RLFT training, during which we need to store the full chains in the buffer, as well as the log probabilities and value functions for each denoising step. 
            This is used to compute the PPO loss's denominator. 
            `sample_mode`: since we need the logprobabilities, we must sample the actions from the sde, hence the hard-coded option "sde". 
            """
            deployable_action_chunk, chains, log_probs, values=self._run_full_inference(batch=batch, sample_mode="sde", denoise_steps=self.config.num_steps, clip_intermediate_actions=self.clip_intermediate_actions, get_chains=True, get_logprob=True, get_value=True) # type: ignore
            return deployable_action_chunk, chains, log_probs, values
        else:
            """
            This option is for RLFT evaluation, during which we only need to sample the final actions. 
            Call the run_inference_full function with the options, and discard everything except the real actions we will directly send to the simulator (real world). 
            `sample_mode`: notice that we can even do ODE inference during evaluation, so we pass the 
            `denoise_steps: we preserve the option to use denoising steps different from RLFT training in evaluation, and it is specified by self.num_steps_eval (default is the same with training). 
            """
            deployable_action_chunk, _, _,_ = self._run_full_inference(batch=batch, sample_mode=self.config.sample_mode, denoise_steps=self.num_steps_eval, clip_intermediate_actions=self.clip_intermediate_actions, get_chains=False, get_logprob=False, get_value=False) # type: ignore
            return deployable_action_chunk, None, None, None
    
    def recalculate_logprob_value_entropy_noise(self,
                                                batch:dict,
                                                chains_pre: Tensor,
                                                chains_next: Tensor,
                                                get_entropy: bool,
                                                get_chains_stds: bool,
                                                clip_intermediate_actions: bool=True)->Tuple[Tensor, Tensor, Tensor|None, Tensor|None]:
        """
        Given the previous and next chains, calculate the joint log probabilities and value functions for each denoising step based on current model parameters,  
        and then do normalizations. This function is used to compute the PPO loss (not for inference) and it records the gradients. 
        
        Args:
            `chains_pre`:  (batch_size, self.config.num_steps, self.config.n_action_steps,self.config.max_action_dim), where self.config.num_steps is the number of denoising steps at training. 
            `chains_next`: (batch_size, self.config.num_steps, self.config.n_action_steps,self.config.max_action_dim) the previous and next denoised actions grouped in batches. chain_next are sampled from the 
            chains_pre under the previous model parameters, and now we want to calculate the log prob at current model parameters to form the new log probs, and the importance sampling term in the PPO loss.
            
            `get_entropy`: bool. Whether to compute the entropy of the denoising chain. 
            `get_chains_stds`: bool. Whether to compute the standard deviation of the denoising chain. 
            `clip_intermediate_actions`: bool. Whether to clip the intermediate actions to the range [self.model.denoise_action_min, self.model.denoise_action_max]
        
        Returns:
            `joint_logprob`: torch.Tensor[batch_size, self.config.act_steps, self.model.effective_action_dim] 
            `value_t_in_batch`: torch.Tensor[batch_size]. 
            `per_symbol_entropy`: torch.Tensor[batch_size, self.config.act_steps, self.model.effective_action_dim]
            `noise_std`: torch.Tensor
        
        Explanation:
            log p(xK|s) = log p(x0)   + \sum_{t=0}^{K-1} log p(xt+1|xt, s)
              H(X0:K)   =  H(x0|s)    + \sum_{t=0}^{K-1}     H(Xt+1|X_t,s)
        """
        # Prepare visual-language-state input
        images, img_masks, lang_tokens, lang_masks, state = self._prepare_visual_language_state_input(batch)
        bsize = state.shape[0]
        device = state.device
        # Represent the visual-language with KV values and then cache them. 
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,  # type: ignore
            past_key_values=None,
            inputs_embeds=[prefix_embs, None], # type: ignore
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        
        # Denoising loop conditions on the Visual-Language cache and recurssively decodes actions by integrating an SDE. 
        n_denoising_steps = self.config.num_steps  # denoising step number at training. 
        x_mean_chain = torch.zeros(bsize, n_denoising_steps, self.model.config.n_action_steps, self.model.config.max_action_dim, device=device) # we use full shape (padded) actions because that is what the model outputs. 
        x_std_chain  = torch.zeros(bsize, n_denoising_steps, self.model.config.n_action_steps, self.model.config.max_action_dim, device=device)
        value_chain  = torch.zeros(bsize, n_denoising_steps, device=device)
        
        # Initial log probability and entropy. 
        x_t = chains_pre[:,0]
        initial_logprob_full = self.model.get_unit_gaussian_noise_log_prob(x_t)
        initial_entropy_full = self.model.get_unit_gaussian_noise_entropy()
        # Denoising loop. 
        for denoise_step in range(n_denoising_steps):
            # Value_t is the value function for each denoising step.  
            x_t = chains_pre[:,denoise_step]
            x_t_plus_one_mean,  x_t_plus_one_std,  value_t = self.model.get_mean_std_val_at_denoise_step(sample_mode="sde",   
                                                                                        clip_intermediate_actions=clip_intermediate_actions,
                                                                                        n_denoise_steps=n_denoising_steps, 
                                                                                        denoise_step_id=denoise_step,
                                                                                        x_t=x_t,
                                                                                        state=state,
                                                                                        prefix_pad_masks=prefix_pad_masks,
                                                                                        past_key_values=past_key_values) # KV for visual-language input
            # Store the mean, std and value for each denoising step. 
            x_mean_chain[:,denoise_step] = x_t_plus_one_mean
            x_std_chain[:,denoise_step] = x_t_plus_one_std
            value_chain[:,denoise_step] = value_t
        # Log transition probability for each prev-next pair in the chains.
        chains_dist = Normal(x_mean_chain,  x_std_chain)
        transition_logprob_full = chains_dist.log_prob(chains_next).sum(dim=1)                                          # sum over denoising steps.        [B, K, Ta, Da] --> [B,Ta,Da]
        joint_logprob_full = initial_logprob_full + transition_logprob_full
        
        # Truncate joint log probability at the effective steps
        joint_logprob = joint_logprob_full[:,:self.config.act_steps,:self.model.effective_action_dim].sum(dim=(-2,-1))  # sum over effective action chunk (open-loop assume independent). [B, Ta, Da] --> [B]
        if self.model.normalize_denoising_horizon:
            joint_logprob = joint_logprob / (1+self.config.num_steps)
        if self.model.normalize_action_chunk_full_dim:
            joint_logprob = joint_logprob / (self.config.act_steps * self.model.effective_action_dim) 
        
        # Decrive per-symbol entropy for the denoising chain.
        if get_entropy:
            transition_entropy_full = chains_dist.entropy().sum(dim=1)                                                   # sum over denoising steps.        [B, K, Ta, Da] --> [B,Ta,Da]
            joint_entropy_full = initial_entropy_full + transition_entropy_full
            joint_entropy=joint_entropy_full[:,:self.config.act_steps,:self.model.effective_action_dim].sum(dim=(-2,-1)) # sum over effective action chunk (open-loop assume independent). [B, Ta, Da] --> [B]
            per_symbol_entropy=joint_entropy/(1+self.config.num_steps)                                                   # by definition . 
            if self.model.normalize_action_chunk_full_dim:
                per_symbol_entropy = per_symbol_entropy / (self.config.act_steps * self.model.effective_action_dim)
        else:
            per_symbol_entropy = None
        
        # Obtain SDE noise statistics by averaging over all denoising steps and action dimensions (for debugging).
        if get_chains_stds:
            noise_std = x_std_chain.mean()
        else:
            noise_std = None 
        
        # Get value function for the final action (precisely speaking, final action's denoising chain) by averaging over denoising steps. 
        value = value_chain.mean(dim=1) # average over denoising steps. 
        
        # Return:
        return joint_logprob, value, per_symbol_entropy, noise_std

    def get_value(self, batch:dict, ignore_value=False):
        """
        Get the value function for the given batch, 
        which will be used to compute the next-step value V(s_t+1) for the truncated actions in the buffer. 
        Warning: this function may need further optimization to be more efficient. 
        Args:
            batch: dict[str, List[str]|Tensor]
            ignore_value: bool. Whether to ignore the value function for simplicity, though at the cost of bringing bias in advantage estimation. 
        Returns:
            value: torch.Tensor[B]
        """
        if ignore_value:
            return 0.0
        
        if not batch:
            raise ValueError(f"batch is empty, cannot get value function.")
        # since we only use this function in the buffer that records training data, 
        # the sample_mode is hard-coded to sde, and the denoise_steps is hard-coded to self.config.num_steps. 
        # the clip_intermediate_actions is variable, which is passed from self.clip_intermediate_actions. 
        # the get_chains and get_logprob are  hard-coded to False to save compute. 
        _, _, _, value = self._run_full_inference(batch, sample_mode="sde", denoise_steps=self.config.num_steps, clip_intermediate_actions=self.clip_intermediate_actions, get_chains=False, get_logprob=False, get_value=True)
        
        return value
    
    def loss(
            self,
            obs,
            chains_pre,
            chains_next,
            returns,
            oldvalues,
            advantages,
            oldlogprobs,
            use_bc_loss=False,
            bc_loss_type: str | None = None,
            clip_intermediate_actions=False,
            verbose=True
            ):
        """
        PPO loss. 
        obs: dict with key state/rgb; more recent obs at the end
        chains_pre: (batch_size, self.config.num_steps, self.config.n_action_steps,self.config.max_action_dim), where self.config.num_steps is the number of denoising steps at training. 
        chains_next: (batch_size, K, self.config.n_action_steps,self.config.max_action_dim)
        returns: (batch_size, )
        values: (batch_size,)
        advantages: (batch_size,)
        oldlogprobs: (batch_size,)
        use_bc_loss: whether to add BC regularization loss
        clip_intermediate_actions: whether to clip the intermediate actions to the range [self.denoise_action_min, self.denoise_action_max]
        """
        
        newlogprobs, newvalues, entropy, noise_std = self.recalculate_logprob_value_entropy_noise(obs,
                                                                                                 chains_pre=chains_pre, 
                                                                                                 chains_next=chains_next, 
                                                                                                 get_entropy=True, 
                                                                                                 get_chains_stds=True,
                                                                                                 clip_intermediate_actions=clip_intermediate_actions) 
        if verbose:
            logger.info(f"oldlogprobs.min={oldlogprobs.min():5.3f}, max={oldlogprobs.max():5.3f}, std of oldlogprobs={oldlogprobs.std():5.3f}")
            logger.info(f"newlogprobs.min={newlogprobs.min():5.3f}, max={newlogprobs.max():5.3f}, std of newlogprobs={newlogprobs.std():5.3f}")
    
        newlogprobs = newlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        oldlogprobs = oldlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        if verbose:
            if oldlogprobs.min() < self.logprob_min: logger.info(f"WARNINIG: old logprobs too low, potential policy collapse detected, should encourage exploration.")
            if newlogprobs.min() < self.logprob_min: logger.info(f"WARNINIG: new logprobs too low, potential policy collapse detected, should encourage exploration.")
            if newlogprobs.max() > self.logprob_max: logger.info(f"WARNINIG: new logprobs too high")
            if oldlogprobs.max() > self.logprob_max: logger.info(f"WARNINIG: old logprobs too high")
        # empirically we noticed that when the min of logprobs gets too negative (say, below -3) or when the std gets larger than 0.5 (usually these two events happen simultaneously) the perfomance drops. 
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if verbose:
            with torch.no_grad():
                advantage_stats = {
                    "mean":f"{advantages.mean().item():2.3f}",
                    "std": f"{advantages.std().item():2.3f}",
                    "max": f"{advantages.max().item():2.3f}",
                    "min": f"{advantages.min().item():2.3f}"
                }
                logger.info(f"Advantage stats: {advantage_stats}")
                corr = torch.corrcoef(torch.stack([advantages, returns]))[0,1].item()
                logger.info(f"Advantage-Reward Correlation: {corr:.2f}")
        
        # Get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()
        
        # Get KL difference estimated via the k3 estimator, ref: http://joschu.net/blog/kl-approx.html). 
        # and how frequently the probability ratio is clipped by the ppo clipping eps. 
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ppo_eps).float().mean().item()

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_ppo_eps, 1 + self.clip_ppo_eps)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        # Value loss
        v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        if self.clip_vloss_coef: # better not use. 
            v_clipped = torch.clamp(newvalues, oldvalues -self.clip_vloss_coef, oldvalues + self.clip_vloss_coef)
            v_loss = 0.5 *torch.max((newvalues - returns) ** 2, (v_clipped - returns) ** 2).mean()
        if verbose:
            with torch.no_grad():
                mse = F.mse_loss(newvalues, returns)
                logger.info(f"Value/Reward alignment: MSE={mse.item():.3f}")
        
        # Entropy loss
        entropy_loss = -entropy.mean() if entropy is not None else 0.0 # if entropy is None, it means that we do not compute entropy, so the entropy loss is 0. 
        # Monitor policy entropy distribution
        if verbose:
            with torch.no_grad():
                logger.info(f"Entropy Percentiles: 10%={entropy.quantile(0.1):.2f}, 50%={entropy.median():.2f}, 90%={entropy.quantile(0.9):.2f}") if entropy is not None else logger.info(f"Entropy is None, cannot monitor entropy distribution.")
        
        # BC loss (optional, not recommended at least for small models.)
        bc_loss = 0.0
        if use_bc_loss:
            raise NotImplementedError(f"bc_loss_type={bc_loss_type} is not implemented.") # TODO: implement BC loss
        
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            bc_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            oldlogprobs.min(),
            oldlogprobs.max(),
            oldlogprobs.std(),
            newlogprobs.min(),
            newlogprobs.max(),
            newlogprobs.std(),
            noise_std.item(),
            # newvalues.mean().item(),#Q function
        )
        