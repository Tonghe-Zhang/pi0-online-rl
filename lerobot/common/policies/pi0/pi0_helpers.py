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
from torch import Tensor
import math 
from lerobot.common.utils.utils import get_safe_dtype
from torch.nn import functional as F
def create_sinusoidal_pos_embedding(
    time: Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
    ) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.
    Args:
      pad_masks is a 2d boolean tensor of shape (batch_size, num_tokens): bool[B, N]. 
        Each item in a batch is true if its part of the real input, false if it is just a padding (e.g., camera not used) and it cannot attend to any other input tokens in the whole input. 
      att_masks is a 2d integer tensor of shape (batch_size, num_tokens): int32[B, N] mask 
        You can view each value 1 in a batch of att_masks as a barrier, that stops the previous tokens from attending to itself and the tokens after it. 
        Examples:
            [[1 1 1 1 1 1]]: pure causal attention. Each token attends to previous tokens and itself, but not those after it. 

            [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
                themselves and the last 3 tokens have a causal attention. The first entry could also be a 1 without changing behaviour.

            [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
                block can attend all previous blocks and all tokens on the same block.
    Returns:
        - att_2d_masks is a 3d tensor of shape (batch_size, num_tokens, num_tokens)
        
    Intermediate variables:
        - att_2d_masks is a 3d tensor of shape (batch_size, num_tokens, num_tokens)
        - pad_2d_masks is a 3d tensor of shape (batch_size, num_tokens, num_tokens)
    
    - About the att_2d_masks: 
        Let's write the cumsum as a matrix c of shape (batch_size, num_tokens). It is the cumulative sum of the att_masks on the columns. 
        If attention masks is [1 0 1 0 1 0 0 1 0 0], then the cumsum is [1 1 2 2 3 3 3 4 4 4]. So this is a stair function. 

        For we write each matrix in cumsum[:, None, :] as c1. Notice that it is row-broadcasted from c, so that c1[i,j]=c[j]. 
        For we write each matrix in cumsum[:, :,None] as c2. Notice that it is column-broadcasted from c, so that c2[i,j]=c[i]. 
        Since the condition att_2d_masks[:,i,j] is true if and only if c1[i,j] <= c2[i,j], this implies it is one only when c[j] <= c[i]. 
        
        Now let us assume i<j. 
        For a stair function cumsum[.], in the same stair (from a '1' in attn_mask to the last index before the next '1'), 
        we have c[i]==c[j], so c[i]>=c[j] holds from left to right or right to left, meaning that there is bidirectional attention within each block. 
        However, if there is a rise in the stair function, meaning that there is a 0->1 change or 1->1 sequence in the att_mask, then c[i] < c[j] for i<j. This means c[j] <= c[i] no longer holds, and 
        therefore the att_2d_masks is zero at those (i,j) pairs. This means those before the rise cannot attend to those after the rise, creating a causal attention mask within blocks. 

        So you can just view the '1' in the att_masks as a barrier that stops the previous tokens from attending to it.
        
        **Each row (i)in the att_2d_masks can see those columns (j) whose index is before it, and those columns whose index is after it and before the next barrier.**
        
    - About the pad_2d_masks: 
        Let's right pad_2d_masks as p, pad_masks[:, None, :] as P1, and aad_masks[:, :, None] as P2, pad_2d_masks as P. 
        P1[i,j]=p[j] and P2[i,j]=p[i]. 
        Then for each matrix in P, P[i,j] is True only when P1[i,j] and P2[i,j] are both True, meaning that p[i]==p[j]==1. 
        
        The att_2d_masks[i,j] is True only when P[i,j] and att_2d_masks are both True, which means that only when none of the i-th and j-th tokens are padded, and they form a causual relation 
        specified by the att_masks. 
        
        In the input pad_masks, a zero in pad_masks[i] means that the i-th token is padded, and therefore the i-th row and i-th column are all zero in the pad_2d_masks, 
        so this token cannot attend to any other token in the entire input. 
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]      # there will be an auto broadcasting when using [:, None, :]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None] # so you are actually comparing two 3d matrices, or batched 2d matrices.
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but got {img.shape}")

    cur_height, cur_width = img.shape[2:]
    # resize image to fit the width and height with bilinear interpolation
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad lines on left+top of image | left   right   top    bottom
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    # the output is of shape (batch_size, channels, width, height)
    return padded_img


def pad_vector(vector, new_dim):
    """
    Input: (batch_size, features) or (batch_size, sequence_length, features)
        (batch_size, features) or (batch_size, sequence_length, features)
    Output: 
        (batch_size, new_dim) or (batch_size, sequence_length, new_dim)

    This function zero-pads a vector along its last dimension to reach a target dimension new_dim. 
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


def get_gaussian_log_prob(sample,mu,sigma):
    """
    The log probability of sample being sampled from a normal distribution with mean mu and variance sigma^2. 
    og N(sample|mu, sigma^2) = ...
    """
    if torch.sum(torch.abs(sigma)) == 0:
        return torch.zeros_like(sample)
    constant_term = -torch.log(sigma) - 0.5 * torch.log(2 * torch.pi * torch.ones_like(sample))
    exponent_term = -0.5 * torch.pow((sample - mu) / sigma, 2)
    log_prob = constant_term + exponent_term
    return log_prob