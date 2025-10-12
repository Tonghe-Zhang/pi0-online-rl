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


import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch

from scripts.env.env_helpers import setup_maniskill_env
from mani_skill.envs.sapien_env import BaseEnv  
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from functools import partial
from scripts.env.multi_action_wrapper import MultiActionWrapper
from scripts.env.per_step_reward_wrapper import PerStepRewardWrapper


env_id="PutOnPlateInScene25Main-v3"   #PutOnPlateInScene25VisionTexture03-v1
n_envs=10
n_steps_episode=1000 
sim_backend="gpu" 
sim_device="cuda" 
sim_device_id=0 
sim_config={"sim_freq": 500, "control_freq": 5} 
sensor_config={"shader_pack": "default"} 
obs_mode="rgb+segmentation" 
control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos" 
episode_mode="train"

wrappers = [
    partial(PerStepRewardWrapper),
    partial(MultiActionWrapper),
]
env: ManiSkillVectorEnv=setup_maniskill_env(env_id, n_envs, n_steps_episode, 
                                            sim_backend, sim_device, sim_device_id, 
                                            sim_config, sensor_config, 
                                            obs_mode, control_mode, 
                                            episode_mode='train', 
                                            wrappers=wrappers)
env_unwrapp: BaseEnv=env.unwrapped # env.reward_mode==None, but it provides episodic success information in info['success']

single_action_dim = env_unwrapp.single_action_space.shape[0] # type: ignore
action_chunk_size=5



print(f"Test env.reset()")
obs, info = env.reset()
print(f"obs={obs.keys()}, info={info.keys()}") # type: ignore


print(f"Test env.step()")
obs, reward, terminated, truncated, info = env.step(torch.randn(n_envs, action_chunk_size, single_action_dim, device=sim_device))
print(f"reward={reward}, terminated={terminated}, truncated={truncated}, info={info.keys()}")






