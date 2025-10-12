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
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
from scripts.env.env_helpers import setup_maniskill_env
from mani_skill.envs.sapien_env import BaseEnv  
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from functools import partial
from scripts.env.multi_action_wrapper import MultiActionWrapper
from scripts.env.per_step_reward_wrapper import PerStepRewardWrapper

# Add imports for YAML and saving images
import yaml
from torchvision.utils import save_image
import torch.nn.functional as F
# Add PIL for text labels
from PIL import Image, ImageDraw, ImageFont




from scripts.env.domain_rand import DomainRandomization
# Load YAML and instantiate with config, replace the existing instantiation
yaml_path = os.path.join(os.path.dirname(__file__), 'domain_rand.yaml')
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)['domain_randomization']
domain_randomizer = DomainRandomization(domain_rand_cfg=config)
print(domain_randomizer)




env_id="PutOnPlateInScene25Main-v3"   #PutOnPlateInScene25VisionTexture03-v1
n_envs=2
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
env_unwrapp: BaseEnv=env.unwrapped 
single_action_dim = env_unwrapp.single_action_space.shape[0] # type: ignore
action_chunk_size=5


obs, info = env.reset()
random_action=torch.randn(n_envs, action_chunk_size, single_action_dim, device=sim_device)
obs, reward, terminated, truncated, info = env.step(random_action)


from scripts.env.env_helpers import fetch_rgb_from_obs
def fetch_batch_from_env(env_unwrapp,obs_venv, model_device, sim_device, proprioception_key, language_key, rgb_keys) -> dict[str, torch.Tensor]:
    """Collect the visual, language, and proprioception observations from the environment. 
    Notice that we have to manually fetch the language from the simulator in case envs reset in the middel of an episode. 
    Returns:
        batch:dict[str, Tensor] 
        batch = {
            proprioception_key: proprioception,                           # Tensor[B, state_dim]. B=num_envs. 
            language_key: language_instruction,                           # A length-B list of strings. 
            rgb_keys[0]:  rgb_image_camera_0                              # Images from camera 0: Tensor[B, C, H, W]
            rgb_keys[1]:  rgb_image_camera_1                              # Images from camera 1: Tensor[B, C, H, W]
            ...
            rgb_keys[num_cameras-1]:  rgb_image_camer_{num_cameras-1}     # Images from the last camera: Tensor[B, C, H, W]
    }
    """
    proprioception: torch.Tensor =env_unwrapp.agent.robot.get_qpos().to(model_device)                       # qpos (joint angles)
    language_instruction: List[str] = env_unwrapp.get_language_instruction()                               # type: ignore a list of strings with len=B=num_envs
    batch = {
        proprioception_key: proprioception,            # Tensor[B, state_dim]. B=num_envs. 
        language_key: language_instruction,            # A length-B list of strings. 
    }
    # add image lists (possibly supporting multiple cameras, including writst camera and 3rd person view camera)
    rgb_image_list=fetch_rgb_from_obs(obs_venv, sim_device, model_device) # type: ignore   # [B, C, H, W], where B here is the num_envs
    for rgb_key, rgb_image in zip(rgb_keys, rgb_image_list):
        batch.update({
            rgb_key: rgb_image                              # Tensor[B, C, H, W], this is transposed from simulator output [B,H,W,C], because PaliGemma image encoder receives images like [B,C,H,W], see code PATH_TO_YOUR_CONDA/envs/pi_r/lib/python3.10/site-packages/transformers/models/paligemma/modeling_paligemma.py function `get_image_features`
        })                        
    return batch

batch = fetch_batch_from_env(env_unwrapp=env_unwrapp, 
                           obs_venv=obs, 
                           model_device=sim_device, 
                           sim_device=sim_device, 
                           proprioception_key="observation.state", 
                           language_key="task", 
                           rgb_keys=["observation.images.top"])

from copy import deepcopy
batch_before_dr=deepcopy(batch)
# Notice that this .apply() is in-place. 
batch_after_dr= domain_randomizer.apply(batch)

# After applying DR, add saving images and checks
# Save images before and after
rgb_key = 'observation.images.top'
save_dir = 'scripts/env/figs/test_domain_rand'
os.makedirs(save_dir, exist_ok=True)

# Save individual pairs
for i in range(n_envs):
    before = batch_before_dr[rgb_key][i]
    after = batch_after_dr[rgb_key][i]
    # Augmentations preserve size, so no resize needed
    pair = torch.cat([before, after], dim=2)  # Horizontal concat
    # save_path = os.path.join(save_dir, f'before_after_{i}.png')
    # save_image(pair, save_path)
    # print(f'Saved pair {i} (before and after) to {save_path}')
    # Compute Frobenius norm difference
    diff_norm = torch.norm(before - after, p='fro')
    print(f"Frobenius norm difference for env {i}: {diff_norm.item()}")
print('Domain randomization applied; check images and norms for differences.')

# Create a tiled grid: 2 rows (before, after), n_envs columns
# We'll need a placeholder for text labels, but since it's not directly supported, we'll just arrange images
# First, ensure all images are the same size
img_h, img_w = batch_before_dr[rgb_key][0].shape[1], batch_before_dr[rgb_key][0].shape[2]
before_imgs = [batch_before_dr[rgb_key][i] for i in range(n_envs)]
after_imgs = [batch_after_dr[rgb_key][i] for i in range(n_envs)]
# Stack into grid: 2 rows, n_envs cols
grid = torch.cat([
    torch.cat(before_imgs, dim=2),  # Concatenate all 'before' horizontally
    torch.cat(after_imgs, dim=2),   # Concatenate all 'after' horizontally
], dim=1)  # Stack the two rows vertically
# save_path_grid = os.path.join(save_dir, 'compare_tiled_all.png')
# save_image(grid, save_path_grid)
# print(f'Saved tiled comparison grid to {save_path_grid} (2 rows: Before, After; columns: Env 0 to {n_envs-1})')

# Use PIL to add labels
# Convert tensor to PIL image
grid_np = grid.permute(1, 2, 0).cpu().numpy() * 255  # Convert to HWC, move to CPU, and scale to 0-255
grid_np = grid_np.astype('uint8')
img = Image.fromarray(grid_np)
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size=20)
except:
    font = ImageFont.load_default()
# Add row labels
label_offset = 10
draw.text((label_offset, img_h // 2), 'Before', fill=(0,0,0), font=font)
draw.text((label_offset, img_h + img_h // 2), 'After', fill=(0,0,0), font=font)
# Add column labels (env_ids)
for i in range(n_envs):
    draw.text((img_w * i + img_w // 2 - 20, label_offset), f'Env {i}', fill=(0,0,0), font=font)
# Save the labeled image
labeled_save_path = os.path.join(save_dir, 'compare_tiled_all_labeled.png')
img.save(labeled_save_path)
print(f'Saved labeled tiled comparison grid to {labeled_save_path} (with Before/After rows and Env IDs on columns)')

# Note: Adding text labels like 'Before', 'After', and env_ids is not directly supported in PyTorch image saving.
# If needed, post-process with PIL or another library outside this script.

# Check if other components are unchanged
print("Proprioception unchanged:", torch.allclose(batch_before_dr['observation.state'], batch_after_dr['observation.state']))
print("Language unchanged:", batch_before_dr['task'] == batch_after_dr['task']) 





