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
from torch import nn
from custom_model.mlp import MLP
from typing import List
class ExploreNoiseNet(nn.Module):
    '''
    Neural network to generate learnable exploration noise, conditioned on time embeddings and or state embeddings. 
    \sigma(s,t) or \sigma(s)
    '''
    def __init__(self,
                 in_dim:int,
                 out_dim:int,
                 hidden_dims:List[int], 
                 activation_type:str,
                 noise_logvar_range:list, #[min_std, max_std]
                 noise_scheduler_type: str
                 ):
        super().__init__()
        self.mlp_logvar = MLP(
            [in_dim] + hidden_dims +[out_dim],
            activation_type=activation_type,
            out_activation_type="Identity",
        )
        self.noise_scheduler_type=noise_scheduler_type
        self.set_noise_range(noise_logvar_range)
    
    def set_noise_range(self, noise_logvar_range:list):
        self.noise_logvar_range=noise_logvar_range
        noise_logvar_min = self.noise_logvar_range[0]
        noise_logvar_max = self.noise_logvar_range[1]
        self.logvar_min = torch.nn.Parameter(torch.log(torch.tensor(noise_logvar_min**2, dtype=torch.float32)), requires_grad=False)
        self.logvar_max = torch.nn.Parameter(torch.log(torch.tensor(noise_logvar_max**2, dtype=torch.float32)), requires_grad=False)
    
    def forward(self, noise_feature:torch.Tensor):
        if 'const' in self.noise_scheduler_type: # const or const_schedule_itr
            # pick the lowest noise level when we use constant noise schedulers. 
            noise_std     = torch.exp(0.5 * self.logvar_min)
        else:
            # use learnable noise level.
            noise_logvar  = self.mlp_logvar(noise_feature)
            noise_std     = self.post_process(noise_logvar)
        return noise_std

    def post_process(self, noise_logvar):
        """
        input:
            torch.Tensor([B, Ta , Da])   log \sigma^2 
        output:
            torch.Tensor([B, Ta, Da]),   sigma, floating point values, bounded in [noise_logvar_min, noise_logvar_max]
        """
        noise_logvar = torch.tanh(noise_logvar)
        noise_logvar = self.logvar_min + (self.logvar_max - self.logvar_min) * (noise_logvar + 1)/2.0
        noise_std = torch.exp(0.5 * noise_logvar)
        return noise_std