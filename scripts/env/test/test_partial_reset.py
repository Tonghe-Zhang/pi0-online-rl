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
Minimal script to test the partial reset functionality of PutOnPlateInScene environments in ManiSkill3. 
Source:
    ManiSkill/mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/put_on_in_scene_multi.py
"""




import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from scripts.evaluate.eval_helpers import tile_images, images_to_video
from scripts.env.env_helpers import setup_maniskill_env
from PIL import Image
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

class PartialResetTest:
    def __init__(self, 
                 env_id, 
                 n_envs, 
                 n_steps_episode, 
                 sim_backend, 
                 sim_device, 
                 sim_device_id, 
                 sim_config,
                 sensor_config, 
                 obs_mode, 
                 control_mode, 
                 episode_mode,
                 seed=0,
                 same_seed_for_all_envs=True
                 ):
        self.env_id = env_id
        self.n_envs = n_envs
        self.n_steps_episode = n_steps_episode
        self.sim_backend = sim_backend
        self.sim_device = sim_device
        self.sim_device_id = sim_device_id
        self.sim_config = sim_config
        self.sensor_config = sensor_config
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        self.episode_mode = episode_mode
        self.seed = seed
        self.same_seed_for_all_envs = same_seed_for_all_envs
        self.step = 0
        self.reset_counter = 0  # Track number of resets for varied seeding
        self.snapshot_paths = []  # Store paths to tiled images for video
    
    def create_env(self):
        self.venv: ManiSkillVectorEnv=setup_maniskill_env(self.env_id, self.n_envs, self.n_steps_episode, 
                                                            self.sim_backend, self.sim_device, self.sim_device_id, 
                                                            self.sim_config, self.sensor_config, 
                                                            self.obs_mode, self.control_mode, 
                                                            episode_mode='train')
        self.env_unwrapp=self.venv.unwrapped # env.reward_mode==None, but it provides episodic success information in info['success']
        self.single_action_dim = self.env_unwrapp.single_action_space.shape[0] # type: ignore

    def reset_env(self, reset_options_venv={}):
        # Use different seeds for each reset to ensure variety
        if "env_idx" in reset_options_venv:
            # Partial reset - use varied seeds
            base_seed = self.seed + self.reset_counter * 1000  # Vary seed for each reset
            env_seeds = base_seed if self.same_seed_for_all_envs else [base_seed+i for i in range(self.n_envs)]
        else:
            # Full reset - use original logic
            env_seeds = self.seed if self.same_seed_for_all_envs else [self.seed+i for i in range(self.n_envs)]
        
        self.reset_counter += 1
        print(f"Reset #{self.reset_counter} at step {self.step} with seed={env_seeds}")
        self.obs_venv, self.info_venv = self.venv.reset(seed=env_seeds, options=reset_options_venv)

    def step_env(self):
        self.action_venv = torch.randn(self.n_envs, self.single_action_dim)
        self.obs_venv, self.info_venv, self.terminated_venv, self.truncated_venv, self.reward_venv = self.venv.step(self.action_venv)
        self.done_venv = self.terminated_venv | self.truncated_venv  # type: ignore
        self.step += 1

    def obs_snapshot(self):
        """
        Store the current observation rgb into a tiled image and save. 
        """
        rgb_tensor: torch.Tensor = self.obs_venv["sensor_data"]["3rd_view_camera"]["rgb"]  # type: ignore
        # print(f"Original tensor shape: {rgb_tensor.shape}")
        
        # Convert to numpy [N, H, W, C] format for tile_images function
        rgb_array = rgb_tensor.cpu().numpy()  # [N, H, W, C] format from ManiSkill
        
        # Ensure values are in [0, 255] range for uint8
        if rgb_array.max() <= 1.0:
            rgb_array = (rgb_array * 255).astype(np.uint8)
        else:
            rgb_array = rgb_array.astype(np.uint8)
        
        # Create tiled image using the same function as eval_helpers.py
        tiled_image = tile_images(rgb_array)  # Returns [H*grid_h, W*grid_w, C]
        
        image = Image.fromarray(tiled_image)

        save_dir=f'{os.path.dirname(os.path.abspath(__file__))}/figs/{self.env_id}/'
        os.makedirs(save_dir, exist_ok=True)
        img_path = f"{save_dir}/obs_snapshot_{self.n_envs}envs_step{self.step}.png"
        image.save(img_path)
        self.snapshot_paths.append(img_path)
        print(f"Saved tiled snapshot with {self.n_envs} environments to {img_path}")

    def save_video(self):
        """
        Combine all saved tiled images into a video using images_to_video from eval_helpers.py
        """
        if not self.snapshot_paths:
            print("No snapshots to create video.")
            return
        # Load images as numpy arrays
        images = [np.array(Image.open(p)) for p in self.snapshot_paths]
        save_dir = os.path.dirname(self.snapshot_paths[0])
        video_path = os.path.join(save_dir, "video.mp4")
        images_to_video(images, save_dir, "video", fps=5, verbose=True) # type: ignore
        print(f"Saved video to {video_path}")

            
if __name__ == "__main__":
    EnvModel = PartialResetTest(env_id="PutOnPlateInScene25Main-v3",   #PutOnPlateInScene25VisionTexture03-v1 PutOnPlateInScene25Single-v1
                                n_envs=4, 
                                n_steps_episode=1000, 
                                sim_backend="gpu", 
                                sim_device="cuda", 
                                sim_device_id=0, 
                                sim_config={"sim_freq": 500, "control_freq": 5}, 
                                sensor_config={"shader_pack": "default"}, 
                                obs_mode="rgb+segmentation", 
                                control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos", 
                                episode_mode="train")
    EnvModel.create_env()
    EnvModel.reset_env()
    
    for step in range(15):
        if step % 5 == 0:
            partial_reset_env_ids=[1,3]
            print(f"Manual partial reset at step {step} for env_idx={partial_reset_env_ids}")
            EnvModel.reset_env(reset_options_venv={"env_idx":partial_reset_env_ids})
            print(f"[Partial reset] env.episode_id={EnvModel.venv.unwrapped.episode_id}") # type: ignore  
        print(f"env.episode_id={EnvModel.venv.unwrapped.episode_id}") # type: ignore
        EnvModel.obs_snapshot()
        EnvModel.step_env()
    
    # After loop, save video
    EnvModel.save_video()

    # # free rollout
    # for step in range(10):
    #     EnvModel.step_env()
    #     print(f"env.episode_id={EnvModel.env.unwrapped.episode_id}") # type: ignore
    #     EnvModel.obs_snapshot()

    # env.reset_env(reset_options_venv={"env_idx":[0]})
    # env.obs_snapshot()
    
    # n_steps_rollout=10
    # for step in tqdm(range(n_steps_rollout)):
    #     env.step_env()
    #     print(f"done_venv={env.done_venv}")
    #     if any(env.done_venv):
    #         print(f"done at step {env.step}")
    #         env.obs_snapshot()
    
    