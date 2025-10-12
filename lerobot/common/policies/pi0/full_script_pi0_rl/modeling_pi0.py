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

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer

from lerobot.common.constants import ACTION, OBS_ROBOT
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.pi0.pi0_helpers import *

####################
# for rl functions
from typing import Literal
from collections import namedtuple
Sample = namedtuple("Sample", "final_actions chains log_probs value")
from typing import Tuple
import logging 
logger = logging.getLogger(__name__)
class PI0Policy(PreTrainedPolicy):
    """Wrapper class around PI0FlowMatching model to train and run inference within LeRobot.
    The .model of PI0Policy is a PI0FlowMatching model.
    """

    config_class = PI0Config
    name = "pi0"

    def __init__(
        self,
        config: PI0Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config: PI0Config = config
        print(f"self.config.device={self.config.device}")
        # In lerobot, the normalize and unnormalize functions 
        # convert raw data into either i) zero-mean unit variance or ii) normalized to [-1, 1]
        # and the unnormalize function reverts this transformation.        
        # In the pi0 standard configuration, the image are not normalized, while the state and action are normalized to zero mean and unit variance.
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        
        
        print(f"config.input_features={config.input_features}")
        print(f"config.normalization_mapping={config.normalization_mapping}") 
        print(f"dataset_stats={dataset_stats}")
        print(f"self.normalize_inputs={self.normalize_inputs}")
        
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        print(f"config.output_features={config.output_features}")
        print(f"self.normalize_targets={self.normalize_targets}")
        
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        self.model = PI0FlowMatching(config)

        if not self.config.act_steps <=self.config.n_action_steps:
            raise ValueError(f"self.config.n_action_steps={self.config.n_action_steps} should be no less than self.config.act_steps={self.config.act_steps}")
        print(f"self.config.n_action_steps={self.config.n_action_steps}, self.config.act_steps={self.config.act_steps}")
        
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters() # type: ignore
    
    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        
        Output shape: (batch_size, action_dim)
        """
        self.eval()

        if self.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])

        batch = self.normalize_inputs(batch)

        # Action queue logic for n_action_steps > 1. 
        # Keep poping out the action queue until it has depleted, then refill it with a new action chunk by all at once by querying the policy.
        if len(self._action_queue) == 0:
            # when the queue is depleted, refill
            images, img_masks = self.prepare_images(batch) # resize, pad, and normalize to [-1,1]
            state = self.prepare_state(batch) # zero-pad to max_state_dim
            lang_tokens, lang_masks = self.prepare_language(batch)

            actions = self.model.sample_actions(
                images, img_masks, 
                lang_tokens, lang_masks, 
                state, 
                noise=noise
            )

            # Unpad actions
            original_action_dim = self.config.action_feature.shape[0]   # type: ignore
            actions = actions[:, :self.config.act_steps, :original_action_dim]   # Revised by Tonghe: we only use the first `act_steps` during execution. 
            # Unnormalize the actions to the original range. 
            actions = self.unnormalize_outputs({"action": actions})["action"]

            if self.config.adapt_to_pi_aloha:
                actions = self._pi_aloha_encode_actions(actions)

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        # When the queue is not depleted or has just been refilled, pop out the oldest action in the queue. 
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor], noise=None, time=None, verbose:bool=False) -> tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass to compute the loss
        
        **Returns**
            loss: torch.Tensor
            loss_dict:dict[str, Tensor]
                loss_dict["losses_after_forward"]
                (optionally) loss_dict["losses_after_in_ep_bound"] 
                loss_dict["losses_after_rm_padding"]
                loss_dict["l2_loss"]
        
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_is_pad")
        """
        images is a list with the length of camera number, and each item is a torch.Size([B, C, 224, 224]), 
        img_masks is a list with the length of camera number, and each item is a torch.Size([B])

        lang_tokens=torch.Size([B, self.config.tokenizer_max_length]), 
        lang_masks=torch.Size([B, self.config.tokenizer_max_length])
        
        state=torch.Size([B, 32])
        actions=torch.Size([B, 32])
        """
        if verbose:
            print(f"images={len(images)} list with {images[0].shape}, img_masks={len(img_masks)} list with {img_masks[0].shape}, content={img_masks[0]}")
            print(f"state={state.shape}")
            print(f"lang_tokens={lang_tokens.shape}, lang_masks={lang_masks.shape}, content={lang_masks}")
            print(f"actions={actions.shape}")
        
        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time, verbose=verbose)
        loss_dict["losses_after_forward"] = losses.clone()
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()
        
        # Remove padding
        if verbose:
            print(f"DEBUG loss padding removal:: lerobot/common/policies/pi0/modeling_pi0.py : losses={losses.shape}, self.config.max_action_dim={self.config.max_action_dim}, self.config.action_feature.shape[0]={self.config.action_feature.shape[0]}")
        losses = losses[:, :, : self.config.max_action_dim]    # should truncate at the original_action_dim = self.config.action_feature.shape[0]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # For backward pass
        loss = losses.mean()
        # For logging
        loss_dict["l2_loss"] = loss.item()

        return loss, loss_dict

    def prepare_images(self, batch, verbose:bool=False):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, 
        convert real pixel inputs from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP. 
        and pad those unused cameras with all black images (all -1 values). 
        
        This function extracts camera names from the keys of `batch`, and applies rescaling and zero-padding to the images to force them display
        in the shape of `self.config.resize_imgs_with_padding`. 
        
        Returns:
            images: a list of length `len(self.config.image_features)`, with each item is a batch of data from that camera, which is 
            of shape [B, C, H, W]. Here, H and W are specified by `self.config.resize_imgs_with_padding`. 
            
            img_masks: a list of length `len(self.config.image_features)`, with each item is a batch of zero-one torch masks from that camera, 
            which is of shape [B,]. Each element in each batch is 1 if the image token is present in the batch and False if it is padded (unused camera). 
        """
        images = []
        img_masks = []
        if self.config.image_features=={}:
            raise ValueError(f"self.config.image_features={self.config.image_features} is empty, but we have batch.keys()={batch.keys()}, check your configuration.")
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        if verbose:
            print(f"self.config.image_features={self.config.image_features}, present_img_keys={present_img_keys}, missing_img_keys={missing_img_keys}")
            print(f"self.config.resize_imgs_with_padding={self.config.resize_imgs_with_padding}")
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images (all black, and it correponds to -1 valued images after shifting the rgb values to [-1,1]). 
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        
        return images, img_masks

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input. 
        
        Returns:
            lang_tokens: [B, self.config.tokenizer_max_length]
            lang_masks:  [B, self.config.tokenizer_max_length]
        """
        device = batch[OBS_ROBOT].device
        tasks = batch["task"]

        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_ROBOT], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions


class PI0FlowMatching(nn.Module):
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
        super().__init__()
        self.config = config

        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_export_config)

        # Projections are float32
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.set_requires_grad()# activate gradients for the state projection matrix.
        
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)
        
        # (Optional) RL functions
        # The additional self.config entries will be appended to the PI0Config before training starts. 
        # Details: 
        # add the value network. 
        if self.config.get("adv_method", None) == "gae":
            self.value_net = nn.Linear(self.config.proj_width, 1)
        
        # add the noise prediction network (optionally) which receives suffix_out decoded from paligemma_with_expert
        self.sde_mode=self.config.get("sde_mode", None)
        if self.sde_mode=='reinflow':
            self.reinflow_noise_scale_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)
        elif self.sde_mode ==None or self.sde_mode not in ['reinflow', 'flow-grpo']:
            logger.warning(f"sde_mode={self.sde_mode} is not supported, methods involving SDE will not function. Please check your config.yaml")
        
        # add the effective action dimension
        if self.config.action_feature is not None:
            self.effective_action_dim = self.config.action_feature.shape[0]  
        else:
            self.effective_action_dim = self.config.max_action_dim
            logger.warning(f"No action feature specified, using max_action_dim={self.effective_action_dim}")
        
        self.denoise_action_min = self.config.denoise_action_min
        self.denoise_action_max = self.config.denoise_action_max
        
    
    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj   # default is true in PI0Config

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, verbose:bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        
        Args:
            images: list of torch.Tensor of shape [B, C, H, W]
            img_masks: list of torch.Tensor of shape [B]
            lang_tokens: torch.Tensor of shape [B, L]
            lang_masks: torch.Tensor of shape [B, L], 
            where B is batch size, 
            C=3 channel number, 
            H and W are specified by self.config.resize_imgs_with_padding
            L=self.config.tokenizer_max_length
            
        Returns:
            embs: torch.Tensor of shape [B, T, D]
            pad_masks: torch.Tensor of shape [B, T]
            att_masks: torch.Tensor of shape [B, T]
            
            where 
            D is the embedding dimension, which is 2048 for the SigLIP vision encoder in paligemma_with_expert. 
            T=M × 256 + L, M is the number of cameras, as specified in self.config.image_features
                256 is the number of image tokens per camera, 
                    256==(H/14)*(W/14)==(224/14)^2=16^2, 
                        224 is determined by self.config.resize_imgs_with_padding. 
                        14 is determined by the SigLIP vision encoder in paligemma_with_expert, and the 
                L=512 is the number of language tokens, specified in self.config.tokenizer_max_length
        """
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []

        # TODO: remove for loop
        for i, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):    # loop over cameras for a total of num_camera times. 
            
            img_emb = self.paligemma_with_expert.embed_image(img)
            """
            print(f"i={i}, img= {img.shape}, img_emb={img_emb.shape}")
                # In PaliGemma's SigLIP vision encoder:
                Input: [4, 3, 224, 224]
                    ↓
                1. Divide into patch, where each path contains 14×14 pixels, and there are in total (224/14)^2=16×16 = 256 patches per image
                2. Each patch → linear projection → 2048-dim embedding, where 2048 is hardcoded in paligemma's configs as the projection_dim. 
                    ↓
                Output: [4, 256, 2048]
            """
            
            img_emb = img_emb.to(dtype=torch.bfloat16)
            
            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)
            
            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs
            
            if verbose:
                print(f"Image embedding i={i}, num_img_embs={num_img_embs}")
            """
            i=0, img=torch.Size([4, 3, 224, 224]), img_emb=torch.Size([4, 256, 2048])
            """
        
        if verbose:
            print(f"$$embs after image embedding: {len(embs)}") # 256
        
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        if verbose:
            print(f"$$lang_emb={lang_emb.shape}")   # 512

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_prefix_fast(
        self, images: list[torch.Tensor], img_masks: list[torch.Tensor], 
        lang_tokens: torch.Tensor, lang_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimized version of embed_prefix that eliminates Python lists and for-loops.
        Embeds images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        
        Args:
            images: List of image tensors [B, C, H, W]
            img_masks: List of image masks [B]
            lang_tokens: Language token tensor [B, L]
            lang_masks: Language mask tensor [B, L]
            
        Returns:
            tuple of:
            - embs: Combined embeddings tensor [B, T, D]
            - pad_masks: Combined padding masks [B, T]
            - att_masks: Combined attention masks [B, T]
        """
        # Get key dimensions
        batch_size = images[0].shape[0]
        device = images[0].device
        dtype = torch.bfloat16
        
        # Stack images and masks for vectorized processing
        stacked_images = torch.stack(images)  # [N, B, C, H, W]
        stacked_masks = torch.stack(img_masks)  # [N, B]
        
        # Get image embeddings for all images at once
        img_embs = self.paligemma_with_expert.embed_image(
            stacked_images.view(-1, *stacked_images.shape[2:])  # [N*B, C, H, W]
        )  # [N*B, P, D], D is the image embedding dimension, and P is the number of image tokens (after tokenization)
        img_embs = img_embs.view(len(images), batch_size, -1, img_embs.shape[-1])  # [N, B, P, D]
        
        # Convert to bfloat16
        img_embs = img_embs.to(dtype=dtype)
        
        # Normalize image embeddings
        img_emb_dim = img_embs.shape[-1]
        img_embs = img_embs * torch.tensor(img_emb_dim**0.5, dtype=dtype, device=device)
        
        # Get language embeddings
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)  # [B, L, D]
        
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        
        # Pre-compute total sequence length
        num_img_tokens_per_img = img_embs.shape[2]  # P
        total_img_tokens = len(images) * num_img_tokens_per_img  # N*P
        num_lang_tokens = lang_emb.shape[1]  # L
        total_seq_len = total_img_tokens + num_lang_tokens  # N*P + L
        
        # Pre-allocate final tensors for embeddings and padding masks of visual and language tokens
        embs = torch.empty(
            (batch_size, total_seq_len, img_emb_dim),
            dtype=dtype,
            device=device
        )
        pad_masks = torch.empty(
            (batch_size, total_seq_len),
            dtype=torch.bool,
            device=device
        )
        
        # Fill embeddings and padding masks
        for i in range(len(images)):
            start_idx = i * num_img_tokens_per_img
            end_idx = start_idx + num_img_tokens_per_img
            embs[:, start_idx:end_idx] = img_embs[i]
            pad_masks[:, start_idx:end_idx] = stacked_masks[i][:, None].expand(-1, num_img_tokens_per_img)
            
        # Add language embeddings and masks at the end
        embs[:, -num_lang_tokens:] = lang_emb
        pad_masks[:, -num_lang_tokens:] = lang_masks
        
        # Create attention masks
        # Images can attend to each other (0s), language tokens can attend to everything (0s)
        att_masks = torch.zeros(total_seq_len, dtype=torch.bool, device=device)
        att_masks = att_masks[None, :].expand(batch_size, -1)
        
        return embs, pad_masks, att_masks

    def embed_suffix(self, state:Tensor, noisy_actions:Tensor, timestep:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing.
        Args:
            state: torch.Tensor of shape [B, state_dim]
            noisy_actions: torch.Tensor of shape [B, self.config.n_action_steps, self.config.max_action_dim]
            timestep: torch.Tensor of shape [B,]
        
        Returns:
            embs: torch.Tensor of shape torch.Size([B, 1+self.config.n_action_steps, self.config.proj_width]) 
            pad_masks: torch.Tensor of shape torch.Size([B, 1+self.config.n_action_steps]), 
            att_masks: torch.Tensor of shape [B, 1+self.config.n_action_steps], [1,1,0,0,0,...0]
            here `1` is for the state token, and `self.config.n_action_steps` is for the action-time tokens. 
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        state_emb = self.state_proj(state)                                  # [B, state_dim] -> [B, self.config.proj_width]
        state_emb = state_emb.to(dtype=torch.bfloat16)  
        embs.append(state_emb[:, None, :])                                  # [B, self.config.proj_width] -> [B, 1, self.config.proj_width]
        
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)  # [B, 1]
        pad_masks.append(state_mask)                                        # [B, 1]

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)                       # [B,] -> [B, self.config.proj_width]

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)             # [B, self.config.n_action_steps, self.config.max_action_dim] -> [B, self.config.n_action_steps, self.config.proj_width]

        time_emb = time_emb[:, None, :].expand_as(action_emb)        # [B, self.config.proj_width] -> [B, self.config.n_action_steps, self.config.proj_width]
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)   # [B, self.config.n_action_steps, self.config.proj_width*2]

        action_time_emb = self.action_time_mlp_in(action_time_emb)   # [B, self.config.n_action_steps, self.config.proj_width*2] -> [B, self.config.n_action_steps, self.config.proj_width]
        action_time_emb = F.silu(action_time_emb)                    # [B, self.config.n_action_steps, self.config.proj_width]  # swish == silu  
        action_time_emb = self.action_time_mlp_out(action_time_emb)  # [B, self.config.n_action_steps, self.config.proj_width] -> [B, self.config.n_action_steps, self.config.proj_width]

        # Add to input tokens
        embs.append(action_time_emb)                                 # [B, 1+self.config.n_action_steps, self.config.proj_width]
        # Generate action_time_mask
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        # Final output
        embs = torch.cat(embs, dim=1)                                # [B, 1+self.config.n_action_steps, self.config.proj_width]
        
        pad_masks = torch.cat(pad_masks, dim=1)                      # [B, 1+self.config.n_action_steps]
        
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks)) # [1, self.config.n_action_steps+1] -> [B, 1+self.config.n_action_steps]
        
        # Shape checking:
        
        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None, verbose:bool=False) -> Tensor:
        """
        Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)
        time=torch.Size([4]), x_t=torch.Size([4, 6, 32]), u_t=torch.Size([4, 6, 32])
        prefix_embs=torch.Size([4, 768, 2048]), prefix_pad_masks=torch.Size([4, 768]), prefix_att_masks=torch.Size([4, 768])
        suffix_embs=torch.Size([4, 7, 1024]), suffix_pad_masks=torch.Size([4, 7]), suffix_att_masks=torch.Size([4, 7])
        pad_masks=torch.Size([4, 775]), att_masks=torch.Size([4, 775])
        position_ids=torch.Size([4, 775]), 
        suffix_out=torch.Size([4, 7, 1024])
        """
        if verbose:
            print(f"** Into {self.__class__.__name__}.forward()")
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)   
        
        time_expanded = time[:, None, None]                             # [B, 1, 1]
        x_t = time_expanded * noise + (1 - time_expanded) * actions     # [B, self.config.n_action_steps, self.config.max_action_dim]
        u_t = noise - actions                                           # [B, self.config.n_action_steps, self.config.max_action_dim]
        if verbose:
            print(f"time={time.shape}, x_t={x_t.shape}, u_t={u_t.shape}")
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, 
            verbose=verbose
        )
        if verbose:
            print(f"prefix_embs={prefix_embs.shape}, prefix_pad_masks={prefix_pad_masks.shape}, prefix_att_masks={prefix_att_masks.shape}")
            # [B, T, paligemma_projection_dim]       [B, self.config.tokenizer_max_length]       [B, self.config.tokenizer_max_length]
            # T = num_cameras x (H/14)(W/14) + self.config.tokenizer_max_length = M x 256 + 512
        
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, time
        )
        if verbose:
            print(f"suffix_embs={suffix_embs.shape}, suffix_pad_masks={suffix_pad_masks.shape}, suffix_att_masks={suffix_att_masks.shape}")
            # [B,1+self.config.n_action_steps, self.config.proj_width]  [B, 1+self.config.n_action_steps]    [B, 1+self.config.n_action_steps]
        
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        if verbose:
            print(f"pad_masks={pad_masks.shape}, att_masks={att_masks.shape}")

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)                  #[B, T+1+self.config.n_action_steps, T+1+self.config.n_action_steps]
        position_ids = torch.cumsum(pad_masks, dim=1) - 1                       #[B, T+1+self.config.n_action_steps]
        if verbose:
            print(f"att_2d_masks={att_2d_masks}, position_ids={position_ids.shape}")
        
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        if verbose:
            print(f"suffix_out={suffix_out.shape}")                     # [B, 1+ self.config.n_action_steps, self.config.proj_width], here `1` is for the state.
        suffix_out = suffix_out[:, -self.config.n_action_steps :]       # [B, self.config.n_action_steps, self.config.proj_width]
        
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)             
        v_t = self.action_out_proj(suffix_out)                          # [B, self.config.n_action_steps, self.config.max_action_dim]
        if verbose:
            print(f"v_t={v_t.shape}")

        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size, n_action_steps, maximum_num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        # in physical intelligence's implemetation they use 1->0 to represent the denoising steps, which follows 
        # diffusion model's convention but is the opposite of what we use in flow matching models. 
        # since their pre-training model is trained in that way, fine-tuning it with SFT or RL has to follow the same convention. 
        # Here, the first time is 1.0:
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        # and the last time should be 0.0. For robustness to numerical errors, we stop interation at -dt/2., so you get the line below:
        while time >= -dt / 2: # robust version of time>=0.0
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep.
        Directly generate an action chunk. (open-loop control within)
        """
        
        suffix_out = self.get_suffix_out(state, prefix_pad_masks, past_key_values, x_t, timestep)
        
        v_t = self.action_out_proj(suffix_out) # there is finally a linear projection to the maximum number of motors MLP. self.action_out_proj=nn.Linear(self.config.proj_width, self.config.max_action_dim)
        return v_t
    
    ################################################################################################
    def get_suffix_out(self,state,prefix_pad_masks,past_key_values,x_t,timestep): 
        """Apply one denoising step of the noise `x_t` at a given timestep. 
        This function returns a representation of (o, a_t, t) where 
        o is the observation (proprioception, visual-language inputs), 
        a_t is the action at time t, 
        and t is the timestep. 
        This is the input of the velocity decoder. 
        #### Output:
        
            suffix_out: torch.Tensor(batchsize, self.config.n_action_steps, self.config.max_action_dim)
        
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)
        
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids, # type: ignore
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs], # type: ignore
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out
    
    ################################################################################################
    # rl functions
    def get_timesteps(self,num_steps,device):
        """
        Derive the timesteps of the diffusion model (flow, but following the timestep convention of diffusions)
        t is the timestep, which ranges from 1, 1-△t, 1-2△t, ..., 1/K, 0, where △t=1/K under uniform time discretization. 
        """
        timesteps = torch.linspace(1, 1 / num_steps, num_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        return timesteps
    
    def get_flowgrpo_noise_level(self, timesteps:torch.Tensor):
        """
        Here we derive the SDE noise levels from Flow-GRPO paper. https://arxiv.org/abs/2505.05470
        
        \sigma_t = a \sqrt{\frac{t}{1-t}}
        Here, 
        a is the noise_level
        t is the timestep, which ranges from 1, 1-△t, 1-2△t, ..., 1/K, 0, where △t=1/K. 
        to avoid division by zero at sigma_1, we approximate the sigma value with the nearest timestep 1-△t.
        when t=0, there is no noise added and we do not need to compute the sigma value.
        """
        noise_level = self.config.noise_level
        sigmas = noise_level * torch.sqrt(
            timesteps / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
        )#         to avoid division by zero at sigma_1, we approximate the sigma value with the nearest timestep 1-△t.
        sigmas = sigmas[:-1] #        when t=0, there is no noise added and we do not need to compute the sigma value.
        return sigmas
    
    def sample_action_chain_prob_value(
        self, images, img_masks, lang_tokens, lang_masks, state, 
        sample_mode: Literal["ode","sde"] = "ode",
        denoise_steps: int = 10,
        clip_intermediate_actions: bool = False,
        get_chains: bool = True,
        get_logprob: bool = False,
        get_value: bool = False
        ) -> Tuple[Tensor, Tensor|None, Tensor|None, Tensor|None]:
        """Description: Sample a chain of latent actions of length `denoise_steps` by solving the ODE/SDE, and compute the log probabilities of the joint distribution. 
        We also compute the values of each latent actions. 
        Args:
            get_chains: whether to compute the chains of latent actions. 
            get_logprob: whether to compute the log probabilities of the joint distribution
            get_value: whether to compute the value functions for each denoised action in the batches. 
                the value here is w.r.t. observation AND all denoised actions. 
        Returns:
                `final_actions`: torch.Tensor[batch_size, n_action_steps, action_dim] 
                `chains`: torch.Tensor[batch_size,denoise_steps + 1,n_action_steps,action_dim] 
                `log_probs`: torch.Tensor[batch_size,denoise_steps,n_action_steps,action_dim] 
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
        # Compute image and language representations
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids, # type: ignore
            past_key_values=None,
            inputs_embeds=[prefix_embs, None], # type: ignore
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        
        # Optionally pre-allocate output tensors to save time. 
        chains = torch.zeros(bsize, denoise_steps + 1, self.config.n_action_steps, self.config.max_action_dim, device=device) if get_chains else None
        log_probs = torch.zeros(bsize, denoise_steps, self.config.n_action_steps, self.config.max_action_dim, device=device) if get_logprob else None
        value_chain = torch.zeros(bsize, denoise_steps, device=device) if get_value else None
        
        # Sample initial state
        x_t = self.sample_noise(actions_shape, device)
        if get_chains:
            chains[:, 0] = x_t  # Store initial state  # type: ignore
        # Denoising loop
        for idx in range(denoise_steps):
            x_t_mean, x_t_std, value_t = self.get_mean_std_val(
                sample_mode,
                clip_intermediate_actions, 
                denoise_steps,
                idx,
                x_t,
                state,
                prefix_pad_masks,
                past_key_values   # visual-language representations
            )
            # When doing SDE inference, sample x_t \sim \mathcal{N}(·|x_t_mean, x_t_std^2) with re-parametrization trick. If doing ODE inference, x_t_std is 0 so it is still fine. 
            unit_noise=self.sample_noise(x_t.shape, device)
            x_t = x_t_mean + x_t_std * unit_noise
            
            # Store results directly in pre-allocated tensors
            if get_chains:
                chains[:, idx + 1] = x_t # type: ignore
            if get_logprob:
                log_prob = get_gaussian_log_prob(x_t, x_t_mean, x_t_std)
                log_probs[:, idx] = log_prob # type: ignore
            if get_value:
                value_chain[:, idx] = value_t # type: ignore
        
        # Average value estimate over denoising steps to obtain the value for the final action. 
        value=value_chain.mean(dim=1) if value_chain is not None else None
        return x_t, chains, log_probs, value
        

    def get_mean_std_val(self,
                        sample_mode: str,
                        clip_intermediate_actions: bool,
                        denoise_steps,
                        idx,
                        x_t: torch.Tensor,
                        state: torch.Tensor,
                        prefix_pad_masks,
                        past_key_values)->Tuple[Tensor,Tensor,Tensor]:
        """Compute the mean, std and value at each denoising step in the input batch. 
        Args:
            sample_mode: "ode" or "sde"
            clip_intermediate_actions: whether to clip the intermediate actions to the range [self.denoise_action_min, self.denoise_action_max]
            denoise_steps: the number of denoising steps. we use this parameter to compute the time discretization scheme. 
            idx: the index of the current denoising step. this will affect the noise level of the SDE with flow-grpo sampling. 
            x_t: denoised actions at different timesteps packed into a batch. torch.Tensor[B, self.config.n_action_steps, self.config.max_action_dim]
            state: the current state
            prefix_pad_masks: the padding masks of the prefix
            past_key_values: the past key values of the paligemma model
        
        Returns:
        `x_t_mean,x_t_std,value_t`:
        the mean, variance and value at each denoising step in the input batch.  
        `value_t`: the value function for each denoised action in the input batch, 
        which conditions on the full observation (visual-language-state) and the current denoising action (o, a_t, t), where t is the denoise step. 
        """
        # parameters 
        bsize = state.shape[0]
        device = state.device
        timesteps= self.get_timesteps(denoise_steps,device)
        # sigma, delta and t_i
        delta = timesteps[idx + 1] - timesteps[idx]
        t_i = timesteps[idx]
        t_input = timesteps[idx].expand(bsize)
        # velocity prediction
        suffix_out = self.get_suffix_out(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            t_input,
        )
        v_t = self.action_out_proj(suffix_out)  # velocities of different denoising steps packed into a batch
        
        # value prediction for each denoise step, grouped in batches. 
        if self.config.adv_method == "gae":
            # average over 
            suffix_out = torch.mean(suffix_out,dim = 1,keepdim=False)   # for each denoising step, average over action chunk
            value_t = self.value_net(suffix_out)[:,0]                  # for each denoising step, average over action chunk. 
        else:
            value_t = torch.zeros((bsize),device=device)
        
        # sampling x_t from N(x_t_mean, x_t_std^2)
        if sample_mode == "ode":
            """
            Flow-ODE inference:
                x_{t+\Delta t} = x_t + v_t \Delta t
            """
            weight_x = 1
            weight_v = 1
            weight_std = torch.tensor(0.0, device=device)   # warning: zero noise may cause numerical instability when calculating the log probability. 
        elif sample_mode == "sde":
            if self.sde_mode=='flow-grpo':
                """
                Flow-GRPO: ODE-SDE conversion.  Eq.12 in https://arxiv.org/abs/2505.05470 
                    $x_{t+\Delta t}=x_t+\left[v_\theta(x_t, t)+\frac{\sigma_t^2}{2 t}(x_t+(1-t) v_\theta(x_t, t))\right] \Delta t+\sigma_t \sqrt{\Delta t} \epsilon$
                """
                sigmas=self.get_flowgrpo_noise_level(timesteps)
                sigma_i = sigmas[idx]
                if isinstance(idx,int):
                    weight_x = 1 + sigma_i**2 / (2 * t_i) * delta
                    weight_v = 1 + sigma_i**2 / (2 * t_i) * (1 - t_i) 
                    weight_std = torch.sqrt(-delta)
                else:
                    weight_x = torch.ones_like(sigma_i) + sigma_i**2 / (2 * t_i) * delta
                    weight_v = torch.ones_like(sigma_i) + sigma_i**2 * (1 - t_i) / (2 * t_i)
                    weight_std = torch.sqrt(-delta)
                    weight_x = weight_x[:,None,None].expand_as(x_t)
                    weight_v = weight_v[:,None,None].expand_as(x_t)
                    weight_std = weight_std[:,None,None].expand_as(x_t)
                    delta = delta[:,None,None].expand_as(x_t)
                    sigma_i = sigma_i[:,None,None].expand_as(x_t)
            elif self.sde_mode == 'reinflow':
                """
                    ReinFlow integration. 
                        x_{t+\Delta t} = x_t + v_t \Delta t + \sigma_t(o, x_t, t) \sqrt{\Delta t} \epsilon
                """
                weight_x=1.0
                weight_v=1.0
                weight_std=self.reinflow_noise_scale_proj(suffix_out) * torch.sqrt(-delta)
            else:
                raise NotImplementedError(f"Unsupported sde_mode={self.sde_mode}")
        else:
            raise NotImplementedError(f"Unsupported sample_mode={sample_mode}")
        
        # the distribution from which to draw samples. 
        x_t_mean =weight_x *  x_t  + weight_v * v_t  * delta    # delta is negative following the convention of physical intelligence's pi-zero codebase. (though it is the opposite of what we use in flow matching models.)
        if clip_intermediate_actions:  # prevent excessively large latent actions wander into OOD points of the model. 
            x_t_mean = x_t_mean.clamp(self.denoise_action_min, self.denoise_action_max)
        
        x_t_std = sigma_i * weight_std
        return x_t_mean,x_t_std,value_t

